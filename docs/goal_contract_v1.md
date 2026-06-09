# Goal Contract v1

> Одна страница. Определяет что такое Goal и как система понимает его состояние.

## 1. Поля

```
Goal
├── id: UUID
├── title: str
├── description: str (optional)
├── goal_type: str          — achievable | continuous | directional | exploratory | meta
├── status: str             — draft | active | done | failed | cancelled | frozen
├── completion_strategy: str — subgoals | metric | manual | binary
├── created_at: datetime
├── updated_at: datetime
├── last_activity_at: datetime
├── parent_goal_id: UUID?   — null для корневых целей
├── is_atomic: bool
└── blockers: list[str]
```

## 2. completion_strategy

| Значение | Когда Goal считается COMPLETED |
|----------|-------------------------------|
| **subgoals** | Все дочерние цели завершены |
| **metric** | Целевая метрика достигла порога |
| **manual** | Когда пользователь явно подтвердил |
| **binary** | Один атомарный результат (is_atomic) |

- Определяется при создании цели.
- Может быть изменён через `mutate`, но это rare action.

## 3. Что считается activity

`last_activity_at` обновляется ТОЛЬКО при следующих событиях:

- **execution_completed** — выполнение execution завершилось
- **artifact_created** — создан новый артефакт
- **subgoal_completed** — дочерняя цель перешла в COMPLETED
- **metric_improved** — значение метрики изменилось в правильную сторону
- **blocker_added** / **blocker_resolved** — изменение блокера

НЕ считается activity:
- `goal_viewed` — просмотр
- `goal_edited` — редактирование полей (кроме activity)
- `comment_added` — комментарий
- `status_changed` — изменение статуса (само по себе)

> Правило: activity = **производство нового результата или изменение условий выполнения**.
> Просмотр, редактирование метаданных и обсуждение — это работа пользователя, не прогресс цели.

## 4. GoalState (производный, не хранимый)

```
if status in (done, failed, cancelled):
    COMPLETED

elif blockers:
    BLOCKED

elif days_since(last_activity_at) > 30:
    ABANDONED

elif days_since(last_activity_at) > 7:
    STALLED

elif days_since(last_activity_at) <= 7 and status == active:
    MOVING

else:
    ACTIVE
```

Пороги STALLED (7d) и ABANDONED (30d) — дефолты.
Переопределяются через `goal_contract.stalled_days` и `goal_contract.abandoned_days` (v2).

## 5. Почему это контракт, а не реализация

1. **last_activity_at теперь определён** — не «от чего-то», а от строгого списка event_type
2. **completion_strategy определяет поведение** — GoalStateCalculator не гадает, как понять что цель готова
3. **GoalState выводится, а не хранится** — нет дублирования статуса
4. **Пороги — числа, не магия** — 7 и 30 дней с возможностью переопределения
5. **Отделение activity от изменений** — пользовательский ввод не считается прогрессом

## 6. Что дальше

1. Добавить `completion_strategy` в модель Goal
2. `GoalActivityService.record_activity()` проверяет event_type против белого списка
3. `GoalStateCalculator` — чистая функция, 6 тестов
4. RiskEngine, PriorityEngine, AIOSStateBuilder — читают GoalState

Весь код уже написан, кроме поля `completion_strategy`. Контракт просто фиксирует,
почему код работает именно так, а не иначе.
