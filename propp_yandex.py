# propp_yandex.py — прогон сказок по функциям Проппа через Yandex AI Studio (OpenAI-совместимый API)
#
# Перед запуском: в терминале `yc init` (Yandex Cloud CLI). Скрипт сам берёт IAM-токен через `yc iam create-token`.
# Нужна роль ai.languageModels.user на каталог.
# Рядом со скриптом должны лежать: папка tales/, propp_gold.csv (из propp_project.zip).
# Запуск: python propp_yandex.py
# Ответы пишутся в cache/propp_yandex.jsonl в том же формате, что и в блокноте:
# положите этот файл в папку cache/ рядом с блокнотом — разделы 8–11 его подхватят.
# Скрипт можно останавливать и перезапускать: готовые ответы берутся из кеша.

import csv, json, os, re, time, subprocess
from pathlib import Path
import requests

# ============================ НАСТРОЙКИ ============================
FOLDER_ID = os.environ.get("YC_FOLDER_ID", "")   # идентификатор каталога Yandex Cloud (или переменная окружения YC_FOLDER_ID)

# ключ (имя в результатах) -> имя модели в каталоге AI Studio
YANDEX_MODELS = {
    "yandexgpt-5.1":       dict(uri="yandexgpt-5.1",       max_tokens=2000),
    "deepseek-v4.1-flash": dict(uri="deepseek-v4.1-flash", max_tokens=20000, extra={"reasoning_effort": "low"}),
}

N_ITER        = 5
TEMPERATURE   = 0.7                 # основной прогон, как в блокноте
TEMP_TALES    = ["72", "70", "67"]  # сказки для проверки температуры
TEMP_LOW      = 0.3                 # температура проверочного прогона (ключ в кеше: <условие>_T03)
CONDITIONS    = ["A_names", "B_defs", "C_fewshot"]
FEWSHOT_TALE  = "64"
EXCLUDE_TALES = ["116"]
PAUSE_SEC     = 1.0                 # пауза между запросами
CACHE_PATH    = Path("cache/propp_yandex.jsonl")
# ===================================================================

URL = "https://llm.api.cloud.yandex.net/v1/chat/completions"

# ---------- данные ----------
GOLD = {}
with open("propp_gold.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["tale_id"] not in EXCLUDE_TALES:
            GOLD[r["tale_id"]] = r
TALES = {tid: Path(f"tales/{tid}.txt").read_text(encoding="utf-8").strip() for tid in GOLD}

def gold_nums(tid):
    return [int(t) for t in GOLD[tid]["gold_num"].split() if t.isdigit()]

def est_tokens(text):
    return int(len(text) / 2.8)

# ---------- промпты: ровно те же, что в блокноте (раздел 4) ----------
FUNCS = {
    8:  ("A",  "Вредительство или недостача",
         "антагонист наносит вред кому-то из семьи (похищает, убивает, изгоняет, околдовывает, подменяет, грозит) "
         "ИЛИ кому-то из семьи чего-то не хватает, хочется что-то иметь (невеста, чудесный предмет, лекарство)"),
    9:  ("B",  "Посредничество",
         "о беде или недостаче сообщают; к герою обращаются с просьбой или приказом, его посылают или отпускают"),
    10: ("C",  "Начинающееся противодействие",
         "искатель соглашается или решается действовать против беды"),
    11: ("↑",  "Отправка", "герой покидает дом"),
    12: ("Д",  "Первая функция дарителя",
         "героя испытывают, выспрашивают, просят о помощи или нападают на него — это подготавливает получение "
         "волшебного средства или помощника"),
    13: ("Г",  "Реакция героя",
         "герой реагирует на действия будущего дарителя: выдерживает или не выдерживает испытание, оказывает "
         "услугу, щадит, отказывает"),
    14: ("Z",  "Получение волшебного средства",
         "в распоряжение героя попадает волшебное средство или помощник: дарят, находят, покупают, появляется "
         "сам, предлагает свои услуги"),
    15: ("R",  "Пространственное перемещение",
         "героя переносят, доставляют или приводят к месту, где находится предмет поисков"),
    16: ("Б",  "Борьба", "герой и антагонист вступают в непосредственную борьбу (бой, состязание)"),
    17: ("К",  "Клеймение", "героя метят: рана, кольцо, полотенце, поцелуй"),
    18: ("П",  "Победа", "антагонист побеждён: в бою, в состязании, убит, изгнан"),
    19: ("Л",  "Ликвидация беды или недостачи",
         "исходная беда или недостача устранена: похищенное добыто, убитый оживлён, заколдованный расколдован"),
    20: ("↓",  "Возвращение", "герой возвращается"),
    21: ("Пр", "Преследование", "героя преследуют: гонятся, летят вдогонку, пытаются убить"),
    22: ("Сп", "Спасение от преследования",
         "герой спасается: бегство, препятствия на пути погони, прятание, превращение"),
    23: ("°",  "Неузнанное прибытие", "герой неузнанным прибывает домой или в другую страну"),
    24: ("Ф",  "Притязания ложного героя", "ложный герой предъявляет необоснованные притязания"),
    25: ("З",  "Трудная задача", "герою предлагают трудную задачу"),
    26: ("Р",  "Решение задачи", "задача решена"),
    27: ("У",  "Узнавание", "героя узнают (по метке, кольцу, решённой задаче)"),
    28: ("О",  "Обличение", "ложный герой или антагонист обличён"),
    29: ("Т",  "Трансфигурация", "герою даётся новый облик"),
    30: ("Н",  "Наказание", "враг наказан"),
    31: ("С",  "Свадьба", "герой вступает в брак и/или воцаряется (или получает награду)"),
}
FUNC_IDS = sorted(FUNCS)
SYM = {k: v[0] for k, v in FUNCS.items()}

TASK = ("Ты — фольклорист. Разметь русскую волшебную сказку по функциям действующих лиц "
        "В. Я. Проппа («Морфология сказки», 1928).")

RULES = '''Правила разметки:
1. Используй только функции из списка выше. Записывай их в том порядке, в котором они происходят в тексте.
2. Подготовительную часть сказки (запреты и их нарушения, выведывание, обман до беды) не размечай.
3. Если эпизод повторяется трижды (три брата, три боя, три испытания), отметь его функции один раз.
4. Если сказка состоит из нескольких ходов (новая беда — новый поиск), продолжай разметку: функции могут повторяться.
5. Если одно событие выполняет две функции, запиши обе подряд.
6. К каждой функции приведи короткую цитату из текста (до 8 слов), где она выражена.

Ответ — только JSON без пояснений:
{"functions": [{"f": <номер функции>, "quote": "<цитата>"}, ...]}'''

def func_list(with_defs):
    lines = []
    for k in FUNC_IDS:
        sym, name, d = FUNCS[k]
        lines.append(f"{k} — {name}" + (f": {d}." if with_defs else ""))
    return "\n".join(lines)

SYSTEM = {
    "A_names": TASK + "\n\nФункции (номер — название):\n" + func_list(False) + "\n\n" + RULES,
    "B_defs":  TASK + "\n\nФункции (номер — название: определение):\n" + func_list(True) + "\n\n" + RULES,
}

# ---------- пример для few-shot: «Гуси-лебеди» по разбору Проппа (с. 105–107) ----------
FEWSHOT_ANSWER = [
    (8,  "Налетели гуси-лебеди, подхватили мальчика"),
    (9,  "Пришла девочка, глядь — братца нету!"),
    (10, "бросилась их догонять"),
    (11, "Выбежала в чистое поле"),
    (12, "Печка, печка, скажи, куда гуси полетели?"),
    (13, "О, у моего батюшки пшеничные не едятся!"),
    (12, "попался ёж; хотела она его толкнуть"),
    (13, "побоялась наколоться"),
    (14, "Вон туда-то! — указал."),
    (15, "Вон туда-то! — указал."),
    (19, "Увидела его сестра, подкралась, схватила и унесла"),
    (20, "схватила и унесла"),
    (21, "а гуси за нею в погоню летят"),
    (22, "Речка её посадила под бережок, гуси пролетели"),
]
FEWSHOT_JSON = json.dumps({"functions": [{"f": f, "quote": q} for f, q in FEWSHOT_ANSWER]}, ensure_ascii=False, separators=(",", ":"))
SYSTEM["C_fewshot"] = (SYSTEM["B_defs"] + "\n\nПример: разметка сказки «Гуси-лебеди» по разбору самого Проппа "
                       "(функции в порядке текста, в quote — фрагмент, где функция выражена):\n" + FEWSHOT_JSON)
print("пример совпадает с эталоном 64:", [f for f, _ in FEWSHOT_ANSWER] == gold_nums(FEWSHOT_TALE))

def skip_combo(cond, tid):
    return cond == "C_fewshot" and tid == FEWSHOT_TALE

def build_messages(cond, tid):
    # system одинаковый для всех сказок → кешируется; 5 итераций одной сказки → кешируется весь промпт
    return [
        {"role": "system", "content": SYSTEM[cond]},
        {"role": "user", "content": "Текст сказки:\n<<<\n" + TALES[tid] + "\n>>>\n\nВерни JSON."},
    ]


# ---------- разбор ответа: тот же, что в блокноте ----------
ROMAN = {"VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
         "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24,
         "XXV": 25, "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30, "XXXI": 31}

def to_func(v):
    if isinstance(v, (int, float)) and int(v) in FUNCS:
        return int(v)
    s = str(v).strip()
    m = re.match(r"^\s*(\d{1,2})\b", s)
    if m and int(m.group(1)) in FUNCS:
        return int(m.group(1))
    if s.upper() in ROMAN:
        return ROMAN[s.upper()]
    return None

def count_invalid(text):
    # сколько элементов с номером вне 8–31 (модель выдумала функцию) — отбрасываются при разборе
    try:
        d = json.loads(text[text.find("{"): text.rfind("}") + 1])
        return sum(1 for it in d.get("functions", []) if to_func(it.get("f") if isinstance(it, dict) else it) is None)
    except Exception:
        return 0

def parse_answer(text):
    # → список (номер, цитата) или None
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    a, b = text.find("{"), text.rfind("}")
    cands = [text[a:b + 1]] if 0 <= a < b else []
    a2, b2 = text.find("["), text.rfind("]")
    if 0 <= a2 < b2:
        cands.append(text[a2:b2 + 1])
    for c in cands:
        try:
            d = json.loads(c)
        except Exception:
            continue
        items = d.get("functions") if isinstance(d, dict) else d
        if not isinstance(items, list):
            continue
        out = []
        for it in items:
            if isinstance(it, dict):
                f = to_func(it.get("f", it.get("function", it.get("id"))))
                q = str(it.get("quote", ""))
            else:
                f, q = to_func(it), ""
            if f is not None:
                out.append((f, q))
        return out
    nums = re.findall(r'"f"\s*:\s*"?(\d{1,2})', text)   # JSON сломан (кавычки внутри цитат) — достаём номера
    if nums:
        return [(int(n), "") for n in nums if int(n) in FUNCS]
    return None


# ---------- IAM-токен (живёт 12 ч, обновляем раз в час) ----------
_token = {"value": None, "t": 0}
def iam_token():
    if not _token["value"] or time.time() - _token["t"] > 3600:
        _token["value"] = subprocess.check_output(["yc", "iam", "create-token"], text=True, shell=False).strip()
        _token["t"] = time.time()
    return _token["value"]

# ---------- кеш ----------
CACHE_PATH.parent.mkdir(exist_ok=True)
CACHE = {}
if CACHE_PATH.exists():
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("raw"):
                CACHE[rec["key"]] = rec

def cache_write(key, raw, usage):
    rec = {"key": key, "raw": raw, "usage": usage, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    CACHE[key] = rec
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- вызов модели ----------
def call(model_key, messages):
    cfg = YANDEX_MODELS[model_key]
    payload = {
        "model": f"gpt://{FOLDER_ID}/{cfg['uri']}",
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": cfg["max_tokens"],
    }
    payload.update(cfg.get("extra", {}))
    for attempt in range(1, 9):
        headers = {"Authorization": f"Bearer {iam_token()}", "OpenAI-Project": FOLDER_ID,
                   "Content-Type": "application/json"}
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=900)
        except requests.RequestException as e:
            print(f"    сеть: {e}; жду {10 * attempt} с"); time.sleep(10 * attempt); continue
        if r.status_code == 200:
            d = r.json()
            text = (d["choices"][0]["message"].get("content") or "").strip()
            u = d.get("usage") or {}
            det = u.get("prompt_tokens_details") or {}
            usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                     "cached_tokens": det.get("cached_tokens", 0) or 0}
            if text:
                return text, usage
            print(f"    пустой ответ (finish_reason={d['choices'][0].get('finish_reason')}), повтор")
            payload["max_tokens"] = int(payload["max_tokens"] * 1.5)
            continue
        if r.status_code in (401, 403):
            _token["value"] = None
            if attempt >= 2:
                raise RuntimeError(f"HTTP {r.status_code}: нет доступа (проверь FOLDER_ID, yc init, роль ai.languageModels.user): {r.text[:500]}")
            continue
        if r.status_code == 400 and "reasoning_effort" in payload:
            print(f"    reasoning_effort не принят, повтор без него: {r.text[:150]}")
            payload.pop("reasoning_effort")
            continue
        if r.status_code in (400, 404):
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
        wait = 20 * attempt
        print(f"    HTTP {r.status_code}: {r.text[:200]} — жду {wait} с")
        time.sleep(wait)
    raise RuntimeError("не удалось получить ответ за 8 попыток")

def ask(model_key, cond, it, tid, tag=""):
    key = f"{model_key}|{cond}{tag}|it{it}|{tid}"
    if key in CACHE:
        return "cache"
    msgs = build_messages(cond, tid)
    raw, u = call(model_key, msgs)
    for _ in range(2):                       # {"functions": []} — сбой, повторяем
        if parse_answer(raw) != []:
            break
        print("    пустой список функций, повтор")
        raw, u = call(model_key, msgs)
    if parse_answer(raw) is None:            # одна попытка починить формат
        print("    JSON не разобран, прошу переформатировать")
        fix = msgs + [{"role": "assistant", "content": raw[:3000]},
                      {"role": "user", "content": 'Верни ТОЛЬКО JSON вида {"functions": [{"f": <номер>, "quote": "<цитата>"}]}.'}]
        raw2, u2 = call(model_key, fix)
        if parse_answer(raw2) is not None:
            raw, u = raw2, u2
    cache_write(key, raw, u)
    p = parse_answer(raw)
    return f"{'НЕ РАЗОБРАН' if p is None else len(p)} функций; эталон {len(gold_nums(tid))}; " \
           f"токены {u.get('prompt_tokens')}+{u.get('completion_tokens')}"

# ---------- прогон ----------
# Порядок: для каждой модели сначала основной прогон (T=0.7, все сказки), потом проверка температуры (T=0.3).
def run(model_key, tales, conds, n_iter, temperature, tag):
    global TEMPERATURE
    TEMPERATURE = temperature
    print(f"\n===== {model_key} | T={temperature} | сказки {tales} | итераций {n_iter} =====", flush=True)
    for tid in tales:
        for cond in conds:
            if cond == "C_fewshot" and tid == FEWSHOT_TALE:
                continue
            for it in range(1, n_iter + 1):
                label = f"{model_key} | {cond}{tag} | it{it} | сказка {tid}"
                try:
                    res = ask(model_key, cond, it, tid, tag)
                    if res == "cache":
                        continue
                    print(f"[{time.strftime('%H:%M')}] [{label}] {res}", flush=True)
                except RuntimeError as e:
                    print(f"[{label}] ✗ {e}", flush=True)
                    if "HTTP 40" in str(e):
                        print("  → модель недоступна, перехожу дальше", flush=True)
                        return
                time.sleep(PAUSE_SEC)

if __name__ == "__main__":
    assert FOLDER_ID, "Задай FOLDER_ID в настройках или переменную окружения YC_FOLDER_ID"
    all_tales = sorted(TALES, key=lambda t: len(TALES[t]))
    MAIN_T = TEMPERATURE
    for model_key in YANDEX_MODELS:
        run(model_key, all_tales, CONDITIONS, N_ITER, MAIN_T, "")          # основной прогон
        run(model_key, TEMP_TALES, CONDITIONS, N_ITER, TEMP_LOW, "_T03")   # проверка температуры
    print("\nГотово. Ответов в", CACHE_PATH, ":", len(CACHE))
