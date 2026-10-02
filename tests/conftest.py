"""Shared fixtures.

Model tests need an exported model folder; they are skipped without one.

    FD_MODEL_DIR       exported folder (default: _export/FRIDA-Decisions)
    FD_EXTRA_CASES     optional JSON list of {"name", "request"} added to the cases
    FD_THREADS         CPU threads for torch and onnxruntime (default 6)

Every test runs on CPU. Measured numbers are printed and written to
`tests/_results/<name>.json`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THREADS = int(os.environ.get("FD_THREADS", "6"))
MODEL_DIR = Path(os.environ.get("FD_MODEL_DIR", ROOT / "_export" / "FRIDA-Decisions"))
RESULTS = ROOT / "tests" / "_results"

INTENTS = [
    ("balance", "узнать баланс счёта"), ("topup", "пополнить баланс"),
    ("tariff_change", "сменить тариф"), ("tariff_info", "узнать условия текущего тарифа"),
    ("roaming", "подключить или настроить роуминг"), ("sim_lost", "заблокировать потерянную SIM-карту"),
    ("sim_new", "получить новую SIM-карту"), ("number_port", "перенести номер от другого оператора"),
    ("internet_slow", "медленный мобильный интернет"), ("no_signal", "нет связи или сигнала"),
    ("sms_fail", "не отправляются SMS"), ("call_fail", "не проходят звонки"),
    ("charge_dispute", "оспорить списание денег"), ("autopay", "настроить автоплатёж"),
    ("invoice", "получить детализацию или счёт"), ("promo", "узнать об акциях и скидках"),
    ("family", "подключить семейный тариф"), ("esim", "перейти на eSIM"),
    ("contract_end", "расторгнуть договор"), ("address_change", "сменить адрес в договоре"),
    ("passport_update", "обновить паспортные данные"), ("app_login", "не получается войти в приложение"),
    ("password_reset", "сбросить пароль от личного кабинета"), ("home_internet", "подключить домашний интернет"),
    ("tv", "подключить телевидение"), ("router", "проблема с роутером"),
    ("speed_test", "проверить скорость интернета"), ("bonus", "потратить бонусные баллы"),
    ("operator", "соединить с живым оператором"), ("complaint", "оставить жалобу на обслуживание"),
    ("number_choice", "выбрать красивый номер"), ("device_buy", "купить смартфон в рассрочку"),
    ("insurance", "застраховать устройство"), ("voicemail", "настроить голосовую почту"),
    ("call_forward", "настроить переадресацию звонков"), ("spam_calls", "блокировать спам-звонки"),
    ("child_control", "родительский контроль"), ("debt", "погасить задолженность"),
    ("refund", "вернуть деньги за услугу"), ("other", "другой вопрос"),
]


def generated_cases() -> list[dict]:
    """Shapes the hand-written requests do not cover: a catalog that splits
    into several rows, and a state longer than the default cut."""
    catalog = {key: text for key, text in INTENTS}
    long_state = ("Добрый день! Пишу по поводу домашнего интернета. Последние две недели "
                  "по вечерам связь рвётся каждые десять-пятнадцать минут, роутер "
                  "перезагружал, кабель проверял, мастер приходил и ничего не нашёл. "
                  "Работаю из дома, созвоны срываются, а абонентская плата списывается "
                  "полностью. Прошу прислать мастера ещё раз и сделать перерасчёт. ") * 5
    return [
        {"name": "generated/intent-catalog-40",
         "request": {"state": "Хочу уйти к вам от другого оператора, но сохранить свой номер. Как это сделать?",
                     "questions": {"intent": {"type": "choice",
                                              "instructions": "Какое намерение у клиента?",
                                              "criteria": catalog}}}},
        {"name": "generated/long-state",
         "request": {"state": long_state,
                     "questions": {
                         "topic": {"type": "choice", "instructions": "Тема обращения?",
                                   "criteria": {"internet": "домашний интернет", "mobile": "мобильная связь",
                                                "tv": "телевидение"}},
                         "refund": {"type": "noul", "instructions": "Клиент просит перерасчёт?"},
                         "mood": {"type": "score", "instructions": "Насколько клиент раздражён?",
                                  "criteria": ["спокоен", "недоволен", "в ярости"]}}}},
    ]


def load_cases() -> list[dict]:
    cases = json.loads((ROOT / "tests" / "data" / "requests.json").read_text(encoding="utf-8"))
    cases += generated_cases()
    extra = os.environ.get("FD_EXTRA_CASES")
    if extra:
        cases += json.loads(Path(extra).read_text(encoding="utf-8"))
    return cases


def flat_margins(response: dict) -> list[float]:
    """Margins in candidate order (question order, then option order)."""
    return [m for per_q in response["margins"].values() for m in per_q.values()]


def record(name: str, payload: dict) -> None:
    """Print a measured result and keep it in tests/_results/."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (RESULTS / f"{name}.json").write_text(text + "\n", encoding="utf-8")
    print(f"\n[{name}] {text}")


@pytest.fixture(scope="session")
def cases() -> list[dict]:
    return load_cases()


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not (MODEL_DIR / "decisions_config.json").exists():
        pytest.skip(f"no exported model at {MODEL_DIR}")
    return MODEL_DIR


@pytest.fixture(scope="session")
def torch_judge(model_dir):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(THREADS)
    from frida_decisions import Judge

    return Judge.from_pretrained(model_dir, device="cpu", dtype=torch.float32)
