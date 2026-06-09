# AIOSState v1

> Центральный артефакт MVP. То, что система возвращает в каждый момент времени.
> Одна страница.

## 1. Определение

**AIOSState — point-in-time снимок того, что происходит в AI-OS.**

Не агрегат. Не дашборд. Не модель данных.
Снимок состояния системы, который пользователь видит на одном экране.

## 2. Структура

```
AIOSState
├── timestamp: datetime
│
├── goals
│   ├── total: int
│   ├── moving: list[GoalSummary]
│   ├── stalled: list[GoalSummary]
│   ├── blocked: list[GoalSummary]
│   ├── completed: list[GoalSummary]
│   └── abandoned: list[GoalSummary]
│
├── risks
│   └── list[RiskItem]
│       ├── title, level, source, detail, goal_id, timestamp
│
├── priorities
│   └── list[PriorityItem]
│       ├── goal_id, title, state, priority_score, reason
│
├── active_executions
│   ├── total: int
│   └── list[ExecutionSummary]
│       ├── execution_id, goal_id, status, started_at, duration_s
│
└── recent_events
    └── list[EventSummary]
        ├── event_type, goal_id, title, timestamp
```

GoalSummary:
```
{ id, title, state, stagnation_days, goal_type, is_atomic }
```

## 3. Источники данных

| Поле | Источник | В v1 |
|------|----------|------|
| goals.* | PostgreSQL, Goal + GoalStateCalculator | ✅ реальные данные |
| risks | RiskEngine (читает goals) | ✅ реальные данные |
| priorities | PriorityEngine (читает goals) | ✅ реальные данные |
| active_executions | Execution Kernel (Zhamlik) или Execution таблица | 🔄 заглушка — `total: 0` |
| recent_events | JournalEntry таблица (Zhamlik) | 🔄 заглушка — `[]` |
| timestamp | datetime.utcnow() | ✅ |

## 4. Кто вычисляет

В v1 — один модуль: **AIOSStateBuilder**.

```
build() {
    1. goals  ← GoalStateCalculator.compute_all()
    2. risks  ← RiskEngine.evaluate_all()
    3. priorities ← PriorityEngine.prioritize()
    4. executions ← ExecutionProvider.get_active()   // заглушка
    5. events ← EventProvider.recent()               // заглушка
    6. return AIOSState
}
```

**GoalStateCalculator, RiskEngine, PriorityEngine** — не отдельные сервисы.
Это внутренние функции AIOSStateBuilder, которые могут быть вынесены в отдельные файлы
для тестирования, но не имеют собственного API, транспорта или lifecycle.

## 5. Кто потребляет

| Потребитель | Что берёт | Как |
|-------------|-----------|-----|
| Cockpit UI  | всё | GET /aios/state → render 5 секций |
| Уведомления | risks | Если появился risk level >= high |
| Система | priorities | Если нужен следующий goal для execution |
| Пользователь | всё | Один экран, zero navigation |

В v1 единственный потребитель — Cockpit UI.

## 6. Частота обновления

- GET /aios/state — вычисляется на каждый запрос.
- Нет кэширования в v1.
- Cost: один SELECT по goals + чистая функция (StateCalculator + Risk + Priority).

## 7. Что исчезает

Из предыдущей архитектуры исчезает:

- ~~Отдельный API у GoalStateCalculator~~ — internal функция
- ~~Отдельный API у RiskEngine~~ — internal функция
- ~~Отдельный API у PriorityEngine~~ — internal функция
- ~~Отдельный эндпоинт /goal-state~~ — только /aios/state
- ~~Cockpit как отдельная логика~~ — Cockpit рендерит AIOSState, не имеет собственной бизнес-логики

Остаётся:

- `GoalStateCalculator` — файл с чистой функцией, тестируется отдельно
- `RiskEngine` — файл с чистой функцией, тестируется отдельно
- `PriorityEngine` — файл с чистой функцией, тестируется отдельно
- `AIOSStateBuilder` — оркестратор, вызывает три функции + читает БД
- `GET /aios/state` — единственная точка входа

## 8. Резюме

AIOSState — единственный публичный контракт MVP.
Всё остальное — implementation detail.

- `goals` — сгруппированные по состоянию, чтобы UI не вычислял
- `risks` — внимание: что требует реакции
- `priorities` — действия: что делать следующим
- `active_executions` — context: что прямо сейчас работает
- `recent_events` — traceability: почему система в таком состоянии

Cockpit UI — это просто render AIOSState.
Никакой логики в UI. Только отображение.
