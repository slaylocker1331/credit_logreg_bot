import asyncio
from types import SimpleNamespace

import pytest

import bot
from model import CATEGORICAL
from tests.conftest import RISKY, SAFE


@pytest.mark.parametrize('text, expected', [
    ('50000', 50000), ('50 000', 50000), ('50,000', 50000), ('$50,000', 50000),
    ('1,234,567.5', 1234567.5), ('8.5', 8.5), ('8,5', 8.5), ('8.5%', 8.5), (' 0 ', 0),
    ('abc', None), ('', None), ('nan', None), ('inf', None),
])
def test_parse_number(text, expected):
    assert bot.parse_number(text) == expected


def test_choice_keyboards_match_handlers_and_model():
    for state, (feature, prefix, labels) in bot.CHOICE_FIELDS.items():
        buttons = [b for row in bot.PROMPTS[state][1].inline_keyboard for b in row]
        assert [b.callback_data for b in buttons] == [f'{prefix}:{v}' for v in CATEGORICAL[feature]]
        assert set(labels) == set(CATEGORICAL[feature])


def test_every_question_is_answered():
    asked = {f for f, *_ in bot.NUMBER_FIELDS.values()} | {f for f, *_ in bot.CHOICE_FIELDS.values()}
    assert asked == set(SAFE)


class FakeMessage:
    def __init__(self, text=None):
        self.text = text
        self.replies = []

    async def reply_text(self, text, reply_markup=None):
        self.replies.append(text)


class FakeQuery:
    def __init__(self, data, message):
        self.data = data
        self.message = message

    async def answer(self):
        pass

    async def edit_message_text(self, text):
        pass


def run_form(answers, model_data):
    context = SimpleNamespace(user_data={'answers': {}}, bot_data={'model': model_data})
    chat = FakeMessage()
    state = bot.AGE
    for _ in bot.ORDER:
        if state in bot.NUMBER_FIELDS:
            feature = bot.NUMBER_FIELDS[state][0]
            update = SimpleNamespace(message=FakeMessage(str(answers[feature])))
            update.message.replies = chat.replies
            state = asyncio.run(bot.number_handler(state)(update, context))
        else:
            feature, prefix, _ = bot.CHOICE_FIELDS[state]
            update = SimpleNamespace(callback_query=FakeQuery(f'{prefix}:{answers[feature]}', chat))
            state = asyncio.run(bot.choice_handler(state)(update, context))
    return state, chat.replies[-1]


@pytest.mark.parametrize('profile, verdict', [(SAFE, 'ОДОБРЯТ'), (RISKY, 'НЕ ОДОБРЯТ')])
def test_full_conversation(model_data, profile, verdict):
    state, result = run_form(profile, model_data)
    assert state == bot.PREDICT
    assert verdict in result.splitlines()[0]


def test_invalid_number_keeps_state():
    context = SimpleNamespace(user_data={'answers': {}}, bot_data={})
    update = SimpleNamespace(message=FakeMessage('150'))
    assert asyncio.run(bot.number_handler(bot.AGE)(update, context)) == bot.AGE
    assert context.user_data['answers'] == {}


@pytest.mark.parametrize('state, value, accepted', [
    (bot.EMP_LENGTH, '6', True), (bot.EMP_LENGTH, '7', False),
    (bot.CRED_HIST, '4', True), (bot.CRED_HIST, '5', False),
])
def test_experience_limited_by_age(state, value, accepted):
    context = SimpleNamespace(user_data={'answers': {'person_age': 20}}, bot_data={})
    update = SimpleNamespace(message=FakeMessage(value))
    next_state = asyncio.run(bot.number_handler(state)(update, context))
    assert (next_state != state) == accepted
    if not accepted:
        assert 'в 20 лет' in update.message.replies[-1]
