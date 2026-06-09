# Goal Runtime v1

> Как Goal работает в MVP AI-OS. Последний контракт перед реализацией.

## 1. Предпосылка

Goal Semantics («Goal = desired_state») — истина системы.
Goal Runtime — прагматичный слой для MVP:

```
вместо desired_state → status + last_activity_at + blockers
вместо current_state  → proxy по последнему событию
вместо gap            → дни с last_activity_at
```

Мы знаем, что это proxy. Мы фиксируем это явно.
В v2 заменим proxy на настоящие desired_state и gap.

## 2. Поля Goal (runtime)

```
Goal
├── id: UUID
├── title: str
├── description: str?
├── goal_type: str            — achievable | continuous | directional | exploratory | meta
├── status: str               — draft | active | done | failed | cancelled | frozen
├── created_at: datetime
├── updated_at: datetime
├── last_activity_at: datetime — см. раздел 4
├── parent_goal_id: UUID?
├── is_atomic: bool
└── blockers: list[str]       — текстовые описания блокеров
```

Нет `completion_strategy` — выводится из goal_type и семантики.
Нет `desired_state` — v2.

## 3. GoalState (производный, не хранимый)

Вычисляется из runtime-полей:

```
if status in (done, failed, cancelled):
    COMPLETED

elif blockers:
    BLOCKED

elif days_since(last_activity_at) > 30:
    ABANDONED

elif days_since(last_activity_at) > 7:
    STALLED

elif last_activity_at < 7 дней и status == active:
    MOVING

else:
    ACTIVE
```

- status = draft/frozen → never попадает в GoalState (исключается из обзора)
- ABANDONED ≠ failed — цель просто не двигалась, может быть возобновлена
- STALLED — первая стадия перед ABANDONED, требует внимания
- BLOCKED перекрывает STALLED (blocker важнее отсутствия движения)

## 4. Что обновляет last_activity_at

`last_activity_at` = timestamp последнего события, которое приближает desired_state.
Для MVP это proxy того, что «цель движется».

Обновляется при:

| Событие | Условие |
|---------|---------|
| execution_completed | execution завершился с результатом |
| artifact_created | создан артефакт, привязанный к goal_id |
| subgoal_completed | дочерняя цель перешла в COMPLETED |
| metric_improved | метрика цели улучшилась (не в v1) |
| blocker_resolved | блокер снят |

НЕ обновляется:

| Событие | Почему |
|---------|--------|
| goal_viewed | просмотр — не производство |
| goal_edited | редактирование метаданных |
| comment_added | обсуждение |
| status_changed | изменение статуса — следствие, не причина |
| subgoal_created | создание подцели — ещё не прогресс |
| blocker_added | добавление блокера — регресс, не прогресс |

Правило: **activity = производство нового результата.**
Просмотр, редактирование, обсуждение — работа пользователя, не прогресс цели.

## 5. Как GoalState попадает в AIOSState

```
Goal (таблица)
    ↓
GoalStateCalculator.compute_all()
    ↓
list[GoalState]  ←  каждый: id, title, state, stagnation_days
    ↓
RiskEngine.evaluate_all_goal_states()
PriorityEngine.prioritize()
    ↓
AIOSState
├── world_state        — (v1: заглушка)
├── goal_progress      — list[GoalState]
├── active_executions  — (v1: заглушка)
├── event_journal      — (v1: заглушка)
└── risks/priorities   — от RiskEngine + PriorityEngine
```

## 6. Жизненный цикл

```
draft → active → [MOVING/STALLED/BLOCKED] → done/failed/cancelled
                 ↓
            ABANDONED (30d без activity)
```

- `draft` → `active`: пользователь или система активирует цель
- `active`: цель в работе, GoalState показывает MOVING/STALLED/BLOCKED
- `done`: цель достигнута (пользователь или completion_strategy)
- `failed`: цель не может быть достигнута
- `cancelled`: цель отменена
- `frozen`: цель заморожена, исключена из обзора

## 7. Границы v1

**Есть в v1:**
- Goal c полями из раздела 2
- GoalStateCalculator на proxy (last_activity_at + status + blockers)
- RiskEngine: STALLED → risk, BLOCKED → risk, ABANDONED → risk
- PriorityEngine: сортировка по состоянию
- AIOSState: 5 секций, goal_progress и risks/priorities — реальные, остальные — заглушки

**Нет в v1:**
- desired_state — proxy вместо него
- current_state provider — нет World Model
- gap function — нет, stagnation_days как proxy gap
- completion_strategy — выводится вручную (пользователь решает, когда done)
- metric_improved — событие зарезервировано, но не реализовано

## 8. Резюме

Goal Runtime v1 — это честный контракт: мы используем proxy, знаем что это proxy,
и фиксируем границу, за которой начинается настоящая семантическая модель (v2).

Весь код, написанный до этого RFC, уже соответствует этому контракту.
Ничего переписывать не нужно.
