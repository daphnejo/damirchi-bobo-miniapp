import base64
import hashlib
import hmac
import json
from decimal import Decimal, InvalidOperation

from django.conf import settings
from django.core.exceptions import ValidationError as DjangoValidationError
from django.utils import timezone
from django_mongodb_backend import transaction

from orders.models import Order
from .models import PaymentTransaction


PAYME_TIMEOUT_MS = 43_200_000  # 12 hours


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def now_ms() -> int:
    return int(timezone.now().timestamp() * 1000)


def get_order(order_id: str) -> Order | None:
    try:
        return Order.objects.get(pk=str(order_id))
    except (Order.DoesNotExist, DjangoValidationError, ValueError, TypeError):
        return None


def mark_order_paid(order: Order) -> bool:
    """Mark an order paid once and report whether this call changed the state."""
    if order.payment_status == Order.PaymentStatus.PAID:
        return False

    order.payment_status = Order.PaymentStatus.PAID
    order.paid_at = timezone.now()
    order.save(update_fields=["payment_status", "paid_at"])
    return True


def notify_paid_order(order: Order) -> None:
    """Best-effort notifications after the payment DB transaction is committed."""
    try:
        from orders.services import (
            send_order_to_operator_group,
            send_payment_confirmation_to_customer,
        )

        send_order_to_operator_group(order)
        send_payment_confirmation_to_customer(order)
    except Exception as exc:
        # Payment confirmation must never be rolled back because Telegram is down.
        print(f"Paid order Telegram notification error: {exc}")


def mark_order_payment_status(order: Order, status: str) -> None:
    if order.payment_status == Order.PaymentStatus.PAID and status != Order.PaymentStatus.REFUNDED:
        return

    order.payment_status = status
    update_fields = ["payment_status"]

    if status != Order.PaymentStatus.PAID and order.paid_at and status != Order.PaymentStatus.REFUNDED:
        order.paid_at = None
        update_fields.append("paid_at")

    order.save(update_fields=update_fields)


# ---------------------------------------------------------------------------
# CLICK Shop API
# ---------------------------------------------------------------------------


CLICK_ERROR_NOTES = {
    0: "Success",
    -1: "SIGN CHECK FAILED",
    -2: "Incorrect parameter amount",
    -3: "Action not found",
    -4: "Already paid",
    -5: "Order not found",
    -6: "Transaction not found",
    -7: "Failed to update order",
    -8: "Error in request",
    -9: "Transaction cancelled",
}


def click_response(payload: dict, error: int, **extra) -> dict:
    response = {
        "click_trans_id": payload.get("click_trans_id", ""),
        "merchant_trans_id": payload.get("merchant_trans_id", ""),
        "error": error,
        "error_note": CLICK_ERROR_NOTES.get(error, "Error"),
    }
    response.update(extra)
    return response


def _click_signature(payload: dict, complete: bool = False) -> str:
    secret_key = str(getattr(settings, "CLICK_SECRET_KEY", ""))

    fields = [
        str(payload.get("click_trans_id", "")),
        str(payload.get("service_id", "")),
        secret_key,
        str(payload.get("merchant_trans_id", "")),
    ]

    if complete:
        fields.append(str(payload.get("merchant_prepare_id", "")))

    fields.extend(
        [
            str(payload.get("amount", "")),
            str(payload.get("action", "")),
            str(payload.get("sign_time", "")),
        ]
    )

    return hashlib.md5("".join(fields).encode("utf-8")).hexdigest()


def _click_signature_valid(payload: dict, complete: bool = False) -> bool:
    actual = str(payload.get("sign_string", "")).lower()
    expected = _click_signature(payload, complete=complete).lower()
    return bool(actual) and hmac.compare_digest(actual, expected)


def _click_amount_matches(order: Order, raw_amount) -> bool:
    try:
        return Decimal(str(raw_amount)) == Decimal(int(order.total_price))
    except (InvalidOperation, TypeError, ValueError):
        return False


def _click_prepare_id(tx: PaymentTransaction) -> str:
    if tx.provider_prepare_id:
        return tx.provider_prepare_id

    # CLICK expects a numeric merchant_prepare_id. MongoDB ObjectId is converted
    # deterministically to a safe 53-bit-ish integer and persisted for retries.
    numeric_id = int(str(tx.pk), 16) % 9_000_000_000_000_000
    tx.provider_prepare_id = str(numeric_id)
    tx.save(update_fields=["provider_prepare_id", "updated_at"])
    return tx.provider_prepare_id


def handle_click_prepare(payload: dict) -> dict:
    if not getattr(settings, "CLICK_ENABLED", False):
        return click_response(payload, -7)

    required = {
        "click_trans_id",
        "service_id",
        "click_paydoc_id",
        "merchant_trans_id",
        "amount",
        "action",
        "error",
        "sign_time",
        "sign_string",
    }
    if any(payload.get(field) in (None, "") for field in required):
        return click_response(payload, -8)

    if str(payload.get("action")) != "0":
        return click_response(payload, -3)

    if str(payload.get("service_id")) != str(getattr(settings, "CLICK_SERVICE_ID", "")):
        return click_response(payload, -8)

    if not _click_signature_valid(payload, complete=False):
        return click_response(payload, -1)

    order = get_order(str(payload.get("merchant_trans_id")))
    if not order or order.payment_type != Order.PaymentType.CLICK:
        return click_response(payload, -5)

    if not _click_amount_matches(order, payload.get("amount")):
        return click_response(payload, -2)

    if order.payment_status == Order.PaymentStatus.PAID:
        return click_response(payload, -4)

    try:
        provider_error = int(payload.get("error", 0))
    except (TypeError, ValueError):
        return click_response(payload, -8)

    if provider_error != 0:
        mark_order_payment_status(order, Order.PaymentStatus.CANCELLED)
        return click_response(payload, -9)

    idempotency_key = f"click:{payload['click_trans_id']}"

    try:
        with transaction.atomic():
            tx = PaymentTransaction.objects.filter(idempotency_key=idempotency_key).first()

            if tx:
                if tx.order_id != order.pk or tx.amount != int(order.total_price):
                    return click_response(payload, -8)

                if tx.status == PaymentTransaction.Status.PAID:
                    return click_response(payload, -4)

                if tx.status in {
                    PaymentTransaction.Status.CANCELLED,
                    PaymentTransaction.Status.FAILED,
                }:
                    return click_response(payload, -9)
            else:
                tx = PaymentTransaction.objects.create(
                    order=order,
                    provider=PaymentTransaction.Provider.CLICK,
                    status=PaymentTransaction.Status.PENDING,
                    amount=int(order.total_price),
                    idempotency_key=idempotency_key,
                    external_id=str(payload["click_trans_id"]),
                    raw_payload=dict(payload),
                )

            prepare_id = _click_prepare_id(tx)

            if order.payment_status != Order.PaymentStatus.PENDING:
                mark_order_payment_status(order, Order.PaymentStatus.PENDING)

        return click_response(
            payload,
            0,
            merchant_prepare_id=int(prepare_id),
        )
    except Exception:
        return click_response(payload, -7)


def handle_click_complete(payload: dict) -> dict:
    if not getattr(settings, "CLICK_ENABLED", False):
        return click_response(payload, -7)

    required = {
        "click_trans_id",
        "service_id",
        "click_paydoc_id",
        "merchant_trans_id",
        "merchant_prepare_id",
        "amount",
        "action",
        "error",
        "sign_time",
        "sign_string",
    }
    if any(payload.get(field) in (None, "") for field in required):
        return click_response(payload, -8)

    if str(payload.get("action")) != "1":
        return click_response(payload, -3)

    if str(payload.get("service_id")) != str(getattr(settings, "CLICK_SERVICE_ID", "")):
        return click_response(payload, -8)

    if not _click_signature_valid(payload, complete=True):
        return click_response(payload, -1)

    order = get_order(str(payload.get("merchant_trans_id")))
    if not order or order.payment_type != Order.PaymentType.CLICK:
        return click_response(payload, -5)

    if not _click_amount_matches(order, payload.get("amount")):
        return click_response(payload, -2)

    tx = PaymentTransaction.objects.filter(
        idempotency_key=f"click:{payload['click_trans_id']}",
        provider=PaymentTransaction.Provider.CLICK,
    ).first()

    if not tx:
        return click_response(payload, -6)

    if str(tx.provider_prepare_id) != str(payload.get("merchant_prepare_id")):
        return click_response(payload, -6)

    # Complete may be retried. Return the same success response for an already
    # completed transaction instead of changing state again.
    if tx.status == PaymentTransaction.Status.PAID and order.payment_status == Order.PaymentStatus.PAID:
        return click_response(
            payload,
            0,
            merchant_confirm_id=int(tx.provider_prepare_id),
        )

    try:
        provider_error = int(payload.get("error", 0))
    except (TypeError, ValueError):
        return click_response(payload, -8)

    try:
        became_paid = False
        with transaction.atomic():
            tx.raw_payload = dict(payload)

            if provider_error != 0:
                tx.status = PaymentTransaction.Status.CANCELLED
                tx.last_error = str(payload.get("error_note") or provider_error)
                tx.save(update_fields=["status", "last_error", "raw_payload", "updated_at"])
                mark_order_payment_status(order, Order.PaymentStatus.CANCELLED)
                return click_response(payload, -9)

            paid_at = timezone.now()
            tx.status = PaymentTransaction.Status.PAID
            tx.paid_at = paid_at
            tx.last_error = None
            tx.save(
                update_fields=[
                    "status",
                    "paid_at",
                    "last_error",
                    "raw_payload",
                    "updated_at",
                ]
            )
            became_paid = mark_order_paid(order)

        if became_paid:
            notify_paid_order(order)

        return click_response(
            payload,
            0,
            merchant_confirm_id=int(tx.provider_prepare_id),
        )
    except Exception:
        return click_response(payload, -7)


# ---------------------------------------------------------------------------
# Payme Merchant API (JSON-RPC 2.0)
# ---------------------------------------------------------------------------


def payme_message(uz: str, ru: str | None = None, en: str | None = None) -> dict:
    return {
        "uz": uz,
        "ru": ru or uz,
        "en": en or ru or uz,
    }


def payme_result(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def payme_error(request_id, code: int, message, data=None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def payme_authorized(authorization_header: str) -> bool:
    login = str(getattr(settings, "PAYME_LOGIN", "")).strip()
    secret = str(getattr(settings, "PAYME_SECRET_KEY", "")).strip()

    if not login or not secret or not authorization_header.startswith("Basic "):
        return False

    try:
        decoded = base64.b64decode(authorization_header[6:].strip()).decode("utf-8")
        supplied_login, supplied_password = decoded.split(":", 1)
    except Exception:
        return False

    return hmac.compare_digest(supplied_login, login) and hmac.compare_digest(
        supplied_password,
        secret,
    )


def _payme_account_error(request_id, field: str):
    return payme_error(
        request_id,
        -31050,
        payme_message(
            "Buyurtma topilmadi.",
            "Заказ не найден.",
            "Order not found.",
        ),
        f"account.{field}",
    )


def _payme_amount_error(request_id):
    return payme_error(
        request_id,
        -31001,
        payme_message(
            "To‘lov summasi noto‘g‘ri.",
            "Неверная сумма.",
            "Incorrect amount.",
        ),
    )


def _payme_operation_error(request_id):
    return payme_error(
        request_id,
        -31008,
        payme_message(
            "Ushbu amalni bajarib bo‘lmaydi.",
            "Невозможно выполнить операцию.",
            "Unable to perform operation.",
        ),
    )


def _payme_transaction_not_found(request_id):
    return payme_error(
        request_id,
        -31003,
        payme_message(
            "Tranzaksiya topilmadi.",
            "Транзакция не найдена.",
            "Transaction not found.",
        ),
    )


def _payme_order_from_params(request_id, params: dict):
    account_field = str(getattr(settings, "PAYME_ACCOUNT_FIELD", "order_id")).strip()
    account = params.get("account")

    if not isinstance(account, dict) or not account.get(account_field):
        return None, _payme_account_error(request_id, account_field)

    order = get_order(str(account.get(account_field)))
    if not order or order.payment_type != Order.PaymentType.PAYME:
        return None, _payme_account_error(request_id, account_field)

    return order, None


def _payme_amount_matches(order: Order, amount) -> bool:
    try:
        return int(amount) == int(order.total_price) * 100
    except (TypeError, ValueError):
        return False


def _payme_tx_result(tx: PaymentTransaction) -> dict:
    return {
        "create_time": int(tx.create_time_ms or 0),
        "perform_time": int(tx.perform_time_ms or 0),
        "cancel_time": int(tx.cancel_time_ms or 0),
        "transaction": str(tx.pk),
        "state": int(tx.provider_state or 0),
        "reason": tx.cancel_reason,
    }


def _payme_statement_item(tx: PaymentTransaction) -> dict:
    data = _payme_tx_result(tx)
    data.update(
        {
            "id": str(tx.external_id or ""),
            "time": int(tx.provider_time_ms or 0),
            "amount": int(tx.amount) * 100,
            "account": tx.account or {},
        }
    )
    return data


def _expire_payme_transaction_if_needed(tx: PaymentTransaction) -> None:
    if tx.provider_state != 1 or not tx.provider_time_ms:
        return

    if now_ms() - int(tx.provider_time_ms) <= PAYME_TIMEOUT_MS:
        return

    tx.provider_state = -1
    tx.status = PaymentTransaction.Status.CANCELLED
    tx.cancel_reason = 4
    tx.cancel_time_ms = now_ms()
    tx.last_error = "Payme transaction timeout"
    tx.save(
        update_fields=[
            "provider_state",
            "status",
            "cancel_reason",
            "cancel_time_ms",
            "last_error",
            "updated_at",
        ]
    )
    mark_order_payment_status(tx.order, Order.PaymentStatus.CANCELLED)


def _payme_duplicate_transaction_error(request_id):
    account_field = str(getattr(settings, "PAYME_ACCOUNT_FIELD", "order_id")).strip() or "order_id"
    return payme_error(
        request_id,
        -31099,
        payme_message(
            "Ushbu buyurtma bo‘yicha boshqa tranzaksiya jarayonda.",
            "По данному заказу уже выполняется другая транзакция.",
            "Another transaction for this order is already in progress.",
        ),
        f"account.{account_field}",
    )


def handle_payme_rpc(payload: dict) -> dict:
    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params")

    if not isinstance(method, str) or not isinstance(params, dict):
        return payme_error(
            request_id,
            -32600,
            payme_message("RPC so‘rovi noto‘g‘ri.", "Неверный RPC-запрос.", "Invalid RPC request."),
        )

    if method == "CheckPerformTransaction":
        order, error = _payme_order_from_params(request_id, params)
        if error:
            return error

        if not _payme_amount_matches(order, params.get("amount")):
            return _payme_amount_error(request_id)

        if order.status == Order.Status.CANCELLED or order.payment_status in {
            Order.PaymentStatus.PAID,
            Order.PaymentStatus.REFUNDED,
        }:
            return _payme_operation_error(request_id)

        return payme_result(request_id, {"allow": True})

    if method == "CreateTransaction":
        order, error = _payme_order_from_params(request_id, params)
        if error:
            return error

        if not _payme_amount_matches(order, params.get("amount")):
            return _payme_amount_error(request_id)

        external_id = str(params.get("id") or "")
        provider_time = params.get("time")
        account = params.get("account") or {}

        if not external_id or provider_time is None:
            return payme_error(
                request_id,
                -32600,
                payme_message("Majburiy maydonlar yetishmaydi.", "Отсутствуют обязательные поля.", "Required fields are missing."),
            )

        try:
            provider_time = int(provider_time)
        except (TypeError, ValueError):
            return payme_error(
                request_id,
                -32600,
                payme_message("time maydoni noto‘g‘ri.", "Неверное поле time.", "Invalid time field."),
            )

        idempotency_key = f"payme:{external_id}"
        existing = PaymentTransaction.objects.filter(idempotency_key=idempotency_key).first()

        if existing:
            _expire_payme_transaction_if_needed(existing)

            if (
                existing.order_id != order.pk
                or existing.amount != int(order.total_price)
                or int(existing.provider_time_ms or 0) != provider_time
                or (existing.account or {}) != account
            ):
                return _payme_operation_error(request_id)

            if existing.provider_state != 1:
                return _payme_operation_error(request_id)

            return payme_result(
                request_id,
                {
                    "create_time": int(existing.create_time_ms or 0),
                    "transaction": str(existing.pk),
                    "state": 1,
                },
            )

        active = PaymentTransaction.objects.filter(
            order=order,
            provider=PaymentTransaction.Provider.PAYME,
            provider_state=1,
        ).first()
        if active:
            _expire_payme_transaction_if_needed(active)
            if active.provider_state == 1:
                return _payme_duplicate_transaction_error(request_id)

        if order.payment_status == Order.PaymentStatus.PAID:
            return _payme_operation_error(request_id)

        try:
            with transaction.atomic():
                created_ms = now_ms()
                tx = PaymentTransaction.objects.create(
                    order=order,
                    provider=PaymentTransaction.Provider.PAYME,
                    status=PaymentTransaction.Status.PENDING,
                    amount=int(order.total_price),
                    idempotency_key=idempotency_key,
                    external_id=external_id,
                    provider_time_ms=provider_time,
                    create_time_ms=created_ms,
                    provider_state=1,
                    account=account,
                    raw_payload=dict(payload),
                )
                mark_order_payment_status(order, Order.PaymentStatus.PENDING)

            return payme_result(
                request_id,
                {
                    "create_time": created_ms,
                    "transaction": str(tx.pk),
                    "state": 1,
                },
            )
        except Exception:
            return payme_error(
                request_id,
                -32400,
                payme_message("Ichki tizim xatosi.", "Системная ошибка.", "System error."),
            )

    if method == "PerformTransaction":
        external_id = str(params.get("id") or "")
        tx = PaymentTransaction.objects.filter(
            idempotency_key=f"payme:{external_id}",
            provider=PaymentTransaction.Provider.PAYME,
        ).first()

        if not tx:
            return _payme_transaction_not_found(request_id)

        _expire_payme_transaction_if_needed(tx)

        if tx.provider_state == 2:
            became_paid = mark_order_paid(tx.order)
            if became_paid:
                notify_paid_order(tx.order)
            return payme_result(
                request_id,
                {
                    "transaction": str(tx.pk),
                    "perform_time": int(tx.perform_time_ms or 0),
                    "state": 2,
                },
            )

        if tx.provider_state != 1:
            return _payme_operation_error(request_id)

        try:
            became_paid = False
            with transaction.atomic():
                performed_ms = now_ms()
                tx.provider_state = 2
                tx.status = PaymentTransaction.Status.PAID
                tx.perform_time_ms = performed_ms
                tx.paid_at = timezone.now()
                tx.raw_payload = dict(payload)
                tx.last_error = None
                tx.save(
                    update_fields=[
                        "provider_state",
                        "status",
                        "perform_time_ms",
                        "paid_at",
                        "raw_payload",
                        "last_error",
                        "updated_at",
                    ]
                )
                became_paid = mark_order_paid(tx.order)

            if became_paid:
                notify_paid_order(tx.order)

            return payme_result(
                request_id,
                {
                    "transaction": str(tx.pk),
                    "perform_time": performed_ms,
                    "state": 2,
                },
            )
        except Exception:
            return payme_error(
                request_id,
                -32400,
                payme_message("Ichki tizim xatosi.", "Системная ошибка.", "System error."),
            )

    if method == "CancelTransaction":
        external_id = str(params.get("id") or "")
        tx = PaymentTransaction.objects.filter(
            idempotency_key=f"payme:{external_id}",
            provider=PaymentTransaction.Provider.PAYME,
        ).first()

        if not tx:
            return _payme_transaction_not_found(request_id)

        if tx.order.status == Order.Status.COMPLETED:
            return payme_error(
                request_id,
                -31007,
                payme_message(
                    "Buyurtma bajarilgan, bekor qilib bo‘lmaydi.",
                    "Заказ выполнен. Невозможно отменить транзакцию.",
                    "Order is completed and cannot be cancelled.",
                ),
            )

        if tx.provider_state in {-1, -2}:
            return payme_result(
                request_id,
                {
                    "transaction": str(tx.pk),
                    "cancel_time": int(tx.cancel_time_ms or 0),
                    "state": int(tx.provider_state),
                },
            )

        if tx.provider_state not in {1, 2}:
            return _payme_operation_error(request_id)

        try:
            reason = int(params.get("reason"))
        except (TypeError, ValueError):
            reason = 10

        try:
            with transaction.atomic():
                cancelled_ms = now_ms()
                was_paid = tx.provider_state == 2
                tx.provider_state = -2 if was_paid else -1
                tx.status = (
                    PaymentTransaction.Status.REFUNDED
                    if was_paid
                    else PaymentTransaction.Status.CANCELLED
                )
                tx.cancel_time_ms = cancelled_ms
                tx.cancel_reason = reason
                tx.raw_payload = dict(payload)
                tx.save(
                    update_fields=[
                        "provider_state",
                        "status",
                        "cancel_time_ms",
                        "cancel_reason",
                        "raw_payload",
                        "updated_at",
                    ]
                )
                mark_order_payment_status(
                    tx.order,
                    Order.PaymentStatus.REFUNDED if was_paid else Order.PaymentStatus.CANCELLED,
                )

            return payme_result(
                request_id,
                {
                    "transaction": str(tx.pk),
                    "cancel_time": cancelled_ms,
                    "state": int(tx.provider_state),
                },
            )
        except Exception:
            return payme_error(
                request_id,
                -32400,
                payme_message("Ichki tizim xatosi.", "Системная ошибка.", "System error."),
            )

    if method == "CheckTransaction":
        external_id = str(params.get("id") or "")
        tx = PaymentTransaction.objects.filter(
            idempotency_key=f"payme:{external_id}",
            provider=PaymentTransaction.Provider.PAYME,
        ).first()

        if not tx:
            return _payme_transaction_not_found(request_id)

        _expire_payme_transaction_if_needed(tx)
        return payme_result(request_id, _payme_tx_result(tx))

    if method == "GetStatement":
        try:
            from_ms = int(params.get("from"))
            to_ms = int(params.get("to"))
        except (TypeError, ValueError):
            return payme_error(
                request_id,
                -32600,
                payme_message("Davr noto‘g‘ri.", "Неверный период.", "Invalid period."),
            )

        transactions = PaymentTransaction.objects.filter(
            provider=PaymentTransaction.Provider.PAYME,
            provider_time_ms__gte=from_ms,
            provider_time_ms__lte=to_ms,
        ).order_by("provider_time_ms")

        return payme_result(
            request_id,
            {"transactions": [_payme_statement_item(tx) for tx in transactions]},
        )

    return payme_error(
        request_id,
        -32601,
        payme_message("Metod topilmadi.", "Метод не найден.", "Method not found."),
        method,
    )


def parse_payme_payload(raw_body: bytes):
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None
