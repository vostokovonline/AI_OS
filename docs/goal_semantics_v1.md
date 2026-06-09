# Goal Semantics v1

> Что такое Goal. Одна страница.

## 1. Определение

**Goal = desired future state of the world**

Цель — это описание желаемого состояния мира. Не задача, не процесс, не набор шагов.
Состояние мира, которое мы хотим получить.

## 2. Термины

```
Goal =
    desired_state

System =
    current_state

gap = distance(desired_state, current_state)

progress = gap(t0) - gap(t1)
         = сокращение разрыва между желаемым и текущим

completion = gap = 0
           = desired_state достигнуто

activity = любое событие, которое уменьшает gap

stalled = gap не уменьшался > N дней

blocked = gap не может уменьшаться из-за внешнего ограничения

risk = gap не уменьшится в ожидаемый срок

priority = какой gap закрывать следующим
```

## 3. Что вытекает

Из этого определения **не нужно решать**, что такое activity, completion, stalled.

- activity — это не «список из 6 event types». Это **любое событие, приближающее desired_state**.
- stalled — это не «7 дней без activity». Это **gap не уменьшается**.
- blocked — это не «поле blockers != []». Это **gap не может уменьшаться**.

Примеры:

| Goal | desired_state | activity |
|------|--------------|----------|
| Запустить MVP | aios_mvp_released = true | коммит, деплой, тест, ревью |
| net_worth >= 100k | net_worth >= 100000 | доход, инвестиция, сокращение расходов |
| Читать каждый день | writing_streak >= 365 | написанный пост |

Activity не нужно определять списком — он выводится из desired_state.

## 4. completion_strategy — производный, не хранимый

`completion_strategy` не нужно хранить в БД. Он выводится из семантики desired_state:

| desired_state | completion_strategy |
|--------------|-------------------|
| Состояние, зависящее от подцелей | subgoals |
| Числовой порог (>=, <=) | metric |
| Булево свойство (true/false) | binary |
| Субъективное решение пользователя | manual |

**Вывод:** `completion_strategy` — это не поле модели.
Это функция от desired_state.

## 5. Что это меняет

1. **Не нужно поле completion_strategy в БД** — оно выводится.
2. **Не нужно поле goal_type для state** — goal_type нужен для UX (как показывать), не для вычислений.
3. **GoalStateCalculator становится семантическим:**
   - gap = 0 → COMPLETED
   - gap frozen > N дней → STALLED
   - gap frozen внешне → BLOCKED
   - gap уменьшается → MOVING
4. **activity не хранится в модели** — `last_activity_at` нужен как proxy для gap, не как список событий.

## 6. Последствия для архитектуры

**Остаётся:**
- `Goal.id`, `title`, `desired_state`, `created_at`, `updated_at`
- `Goal.last_activity_at` — proxy для движения gap
- `Goal.blockers` — внешние ограничения

**Уходит из модели:**
- `completion_strategy` — выводится из desired_state
- `goal_type` — остаётся для UX, не для вычислений

**Меняется:**
- `GoalStateCalculator` — сравнивает desired_state с current_state, не статусы
- `completion` = gap = 0, не status = done

## 7. Открытые вопросы

1. **Как представить desired_state в коде?**
   - JSON? DSL? Ссылка на метрику?
   - Минимум: `{"net_worth": {"gte": 100000}}` — JSON-условие.

2. **Как измерять current_state?**
   - Для метрик: числовое значение.
   - Для булевых: true/false.
   - Для подцелей: статусы дочерних целей.

3. **Как считать gap?**
   - Для числовых: max(0, target - current).
   - Для булевых: 0 или 1.
   - Для подцелей: count(completed) / count(total).

4. **Нужен ли desired_state в v1?**
   - Или достаточно proxy (last_activity_at + status + blockers)?
   - MVP может работать на proxy, но контракт должен быть ясен.

## 8. Резюме

**Goal = desired_state.**

Всё остальное — производные:
- gap, progress, completion, stalled, blocked, risk, priority —
всё это функции от distance(desired_state, current_state).

Без этого контракта любой GoalStateCalculator —
это хардкод, маскирующийся под архитектуру.
