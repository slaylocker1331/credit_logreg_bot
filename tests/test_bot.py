import asyncio
from types import SimpleNamespace

import pytest
from telegram.ext import ApplicationHandlerStop

import bot
from db import Database
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


USER = SimpleNamespace(id=42, username='alice', first_name='Alice', last_name=None, language_code='ru')


def run_form(answers, model_data, db=None, user=USER):
    context = SimpleNamespace(user_data={'answers': {}}, bot_data={'model': model_data, 'db': db})
    chat = FakeMessage()
    state = bot.AGE
    for _ in bot.ORDER:
        if state in bot.NUMBER_FIELDS:
            feature = bot.NUMBER_FIELDS[state][0]
            update = SimpleNamespace(message=FakeMessage(str(answers[feature])), effective_user=user)
            update.message.replies = chat.replies
            state = asyncio.run(bot.number_handler(state)(update, context))
        else:
            feature, prefix, _ = bot.CHOICE_FIELDS[state]
            update = SimpleNamespace(callback_query=FakeQuery(f'{prefix}:{answers[feature]}', chat),
                                     effective_user=user)
            state = asyncio.run(bot.choice_handler(state)(update, context))
    return state, chat.replies


@pytest.mark.parametrize('profile, verdict', [(SAFE, 'ОДОБРЯТ'), (RISKY, 'НЕ ОДОБРЯТ')])
def test_full_conversation(model_data, profile, verdict):
    state, replies = run_form(profile, model_data)
    assert state == bot.PREDICT
    assert verdict in replies[-1].splitlines()[0]


def test_application_saved_to_db(model_data, tmp_path):
    db = Database(str(tmp_path / 'bot.db'))
    run_form(SAFE, model_data, db)
    run_form(RISKY, model_data, db)

    user = db.get_user(USER.id)
    assert user['link'] == 'https://t.me/alice'
    assert user['first_name'] == 'Alice'
    assert user['phone'] is None

    apps = db.get_applications(USER.id)
    assert [a['approved'] for a in apps] == [1, 0]
    assert apps[0]['loan_grade'] == 'A' and apps[0]['person_income'] == 90_000
    assert 0 <= apps[1]['risk'] <= 1


def test_phone_prompt_only_until_shared(model_data, tmp_path):
    db = Database(str(tmp_path / 'bot.db'))
    _, replies = run_form(SAFE, model_data, db)
    assert 'номер телефона' in replies[-1]

    db.save_phone(USER, '+79990000000')
    assert db.get_user(USER.id)['phone'] == '+79990000000'
    _, replies = run_form(SAFE, model_data, db)
    assert 'ОДОБРЯТ' in replies[-1]  # result is the last message, no phone prompt


def test_user_without_username_gets_id_link(tmp_path):
    db = Database(str(tmp_path / 'bot.db'))
    db.upsert_user(SimpleNamespace(id=7, username=None, first_name='Bob', last_name='B', language_code=None))
    assert db.get_user(7)['link'] == 'tg://user?id=7'


@pytest.mark.parametrize('contact_user_id, saved', [(USER.id, True), (999, False)])
def test_only_own_contact_is_saved(tmp_path, contact_user_id, saved):
    db = Database(str(tmp_path / 'bot.db'))
    message = FakeMessage()
    message.contact = SimpleNamespace(user_id=contact_user_id, phone_number='+79990000000')
    update = SimpleNamespace(message=message, effective_user=USER)
    context = SimpleNamespace(bot_data={'db': db})
    with pytest.raises(ApplicationHandlerStop):
        asyncio.run(bot.save_contact(update, context))
    user = db.get_user(USER.id)
    assert (user is not None and user['phone'] == '+79990000000') == saved


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
    update = SimpleNamespace(message=FakeMessage(value), effective_user=USER)
    next_state = asyncio.run(bot.number_handler(state)(update, context))
    assert (next_state != state) == accepted
    if not accepted:
        assert 'в 20 лет' in update.message.replies[-1]
