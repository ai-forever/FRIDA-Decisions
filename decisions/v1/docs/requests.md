# Requests and responses

A request is a `state` — a string or any JSON value (objects are rendered as compact, key-sorted JSON) — and named `questions`. Each question has a `type`, free-text `instructions` and, depending on the type, `criteria`. Every response carries `answers` (one per question), the raw `margins` of every option, and `usage`.

The answers below are real outputs of the released model (float32, CPU), trimmed to three decimals.

### `choice` — pick one of several described options

```json
{"state": "Здравствуйте! Вчера оплатил заказ №5512 картой, деньги списались дважды. Верните, пожалуйста, лишнее списание.",
 "questions": {"topic": {"type": "choice", "instructions": "Какая тема обращения?",
   "criteria": {"billing": "оплата, списания, возвраты денег",
                "delivery": "доставка и сроки получения заказа",
                "account": "вход в аккаунт, пароль, личные данные",
                "product": "качество и характеристики товара"}}}}
```
```json
{"answers": {"topic": {"type": "choice", "probabilities": {"billing": 0.999, "delivery": 0.001, "account": 0.0, "product": 0.001}, "confidence": 0.992, "choice": "billing"}}}
```

An option is shown to the model as `"<id>: <description>"`, so keep ids short and neutral and put the meaning in the description.

### `score` — place the input on an ordered scale

```json
{"state": "Сервер с базой заказов не отвечает уже сорок минут, клиенты не могут оформить покупку, каждая минута — потерянные продажи.",
 "questions": {"urgency": {"type": "score", "instructions": "Оцени срочность обращения по шкале ниже.",
   "criteria": ["Очень низкая: работа не страдает, ответить можно когда угодно.",
                "Низкая: неудобство небольшое, ответить можно на этой неделе.",
                "Средняя: работа затруднена, но обходной путь есть.",
                "Высокая: работа заблокирована, обходного пути нет.",
                "Критическая: потери растут с каждым часом."]}}}
```
```json
{"answers": {"urgency": {"type": "score", "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.086, "4": 0.914}, "confidence": 0.816, "score": 3.913, "legend": {"0": "Очень низкая: …", "...": "…", "4": "Критическая: …"}}}}
```

`score` is the expected level (0 … N−1) under `probabilities`; `legend` maps levels back to their descriptions.

### `noul` — yes / no

```json
{"state": "Курьер опоздал на два часа и даже не позвонил. Больше заказывать у вас не буду.",
 "questions": {"complaint": {"type": "noul", "instructions": "Это жалоба?"}}}
```
```json
{"answers": {"complaint": {"type": "noul", "noul": 0.674}}}
```

`noul` is the probability of "yes". `criteria` is optional: `{"true": "...", "false": "..."}` replaces the default wording of either side.

### `ranking` — order candidates best first

```json
{"state": "Как удалить накипь из чайника?",
 "questions": {"best": {"type": "ranking", "instructions": "Какой фрагмент лучше всего отвечает на вопрос?",
   "criteria": {"p1": "Налейте в чайник воду с двумя ложками лимонной кислоты, вскипятите и оставьте на час, затем промойте.",
                "p2": "Электрический чайник нельзя погружать в воду целиком: это опасно для нагревательного элемента.",
                "p3": "Чай лучше заваривать водой температурой 85–95 градусов, а не крутым кипятком.",
                "p4": "Уксус тоже растворяет накипь: смешайте его с водой один к одному, прокипятите и тщательно сполосните."}}}}
```
```json
{"answers": {"best": {"type": "ranking", "ranking": ["p4", "p1", "p2", "p3"], "scores": {"p1": 4.408, "p2": 0.167, "p3": -0.065, "p4": 5.851}, "probabilities": {"p1": 0.19, "p2": 0.003, "p3": 0.002, "p4": 0.805}, "confidence": 0.625}}}
```

A ranking candidate is content (a passage, an agent action, a document), not a labelled option: its id is never shown to the model. `criteria` may also be a plain list; ids are then `"0"`, `"1"`, … `scores` are the raw margins — use them for top-k or a threshold (see [Filtering ranking candidates by a threshold](#filtering-ranking-candidates-by-a-threshold)); `probabilities` split one unit among the candidates.

### Filtering ranking candidates by a threshold

`probabilities` cannot filter: they split one unit among the candidates, so with 250 candidates a confident winner still reads 0.09. `scores` can — each margin is computed independently of the other candidates, and adding or removing a candidate leaves the rest untouched. What they are not is calibrated: the head is trained with a listwise softmax, which depends only on differences inside one list, so nothing pins the level of a particular request's margins. On the relevance task of razvilka (50 queries, 5 candidates, one relevant and four annotated negatives) that level runs from −1.88 to +10.84 with sd 2.32, against a median gap of +4.92 between the relevant candidate and the best irrelevant one. Everything below is `tools/ranking_threshold.py`; the held-out column fits the threshold on four fifths of the queries and scores it on the rest.

| threshold on | AUC (95 % CI) | best single threshold | held-out F1 |
|---|---|---|---|
| `scores` as they are | 0.923 (0.882–0.959) | +6.76 → F1 0.769 | 0.752 |
| `scores` minus a fixed anchor candidate | 0.929 (0.890–0.964) | +7.18 → F1 0.774 | 0.732 |
| `scores` minus the **median** of the request's own candidates | 0.951 (0.910–0.982) | +4.40 → F1 0.835 | 0.796 |
| `scores` minus the **mean** of the request's own candidates | 0.972 (0.943–0.992) | +3.52 → F1 0.882 | 0.855 |

So subtract the request's own list, not a constant:

```python
answer = judge(request)["answers"]["best"]
scores = answer["scores"]
level = sum(scores.values()) / len(scores)
relevant = [key for key, score in scores.items() if score - level >= 3.5]
```

Adding an anchor candidate — one extra candidate whose text says it is not an answer — is the obvious thing to try and it does not work: its own margin barely moves with the request (median +0.63, sd 0.65, correlation with the request's level +0.22), so subtracting it removes noise rather than the offset. The list does carry the offset, because most of a retriever's top-k is irrelevant and their margins are where that request's "irrelevant" sits.

Which is also what the recipe assumes, and what it cannot do:

* **The threshold is not a constant.** +3.5 is fitted on lists of five with one relevant candidate. Keep it and shorten the list and recall falls away — on lists of three, F1 0.837; on lists of two, 0.689 — because the relevant candidate is itself part of the mean and drags it up. Refitted, the same lists reach 0.917 and 0.953 at +1.62 and −0.09. Re-fit on lists the shape of yours.
* **It cannot say that nothing is relevant.** Drop the relevant candidate and centre the four negatives among themselves, and 5.5 % of them still clear +3.52 (the raw threshold +6.76 leaves 7.0 %): centring ranks the best of the list, it does not recognise relevance. For an absolute per-candidate probability, ask `noul` per candidate — it is trained with a pairwise objective and returns a probability in 0…1.
* **Lists with several relevant candidates are not measured here.** This task has exactly one, so these numbers describe a pool that is 20 % relevant. 50 queries is also a small set: the intervals above are bootstrap over queries, and the ones for the raw and the centred threshold do overlap at the edges.

Invalid requests raise `frida_decisions.RequestError`; `error.payload()` gives `{"error": {"type", "message", "field"}}`.
