"""
Telegram-бот для прогноза одобрения кредита на основе логистической регрессии
"""
import logging
import math
import os
import re
import warnings

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes, ConversationHandler,
    MessageHandler, filters,
)
from telegram.warnings import PTBUserWarning

from model import (
    DEFAULT_ON_FILE, HOME_OWNERSHIP, LOAN_GRADE, LOAN_INTENT, MODEL_PATH, load_model, predict_risk,
)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
# Mixing message and callback handlers in one conversation is intended here
warnings.filterwarnings('ignore', message=r".*CallbackQueryHandler", category=PTBUserWarning)

(AGE, INCOME, HOME, EMP_LENGTH, CRED_HIST, DEFAULT, INTENT, GRADE,
 LOAN_AMOUNT, LOAN_RATE, PREDICT) = range(11)
ORDER = [AGE, INCOME, HOME, EMP_LENGTH, CRED_HIST, DEFAULT, INTENT, GRADE, LOAN_AMOUNT, LOAN_RATE]

HOME_LABELS = {'RENT': 'Аренда', 'OWN': 'Собственное жильё', 'MORTGAGE': 'Ипотека', 'OTHER': 'Другое'}
INTENT_LABELS = {
    'PERSONAL': 'Личные нужды', 'EDUCATION': 'Образование', 'MEDICAL': 'Лечение',
    'VENTURE': 'Бизнес', 'HOMEIMPROVEMENT': 'Ремонт жилья', 'DEBTCONSOLIDATION': 'Погашение других долгов',
}
GRADE_LABELS = {g: g for g in LOAN_GRADE} | {'A': 'A (низкий риск)', 'G': 'G (высокий риск)'}
DEFAULT_LABELS = {'N': 'Нет', 'Y': 'Да'}


def keyboard(prefix: str, options: list[str], labels: dict, per_row: int = 1) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(labels[o], callback_data=f'{prefix}:{o}') for o in options]
    return InlineKeyboardMarkup([buttons[i:i + per_row] for i in range(0, len(buttons), per_row)])


PROMPTS = {
    AGE: ('🎂 Сколько вам лет?', None),
    INCOME: ('💵 Ваш годовой доход (в долларах)?', None),
    HOME: ('🏠 Какой у вас тип жилья?', keyboard('home', HOME_OWNERSHIP, HOME_LABELS)),
    EMP_LENGTH: ('💼 Сколько лет вы работаете на текущем месте? (можно 0)', None),
    CRED_HIST: ('📅 Сколько лет вашей кредитной истории?', None),
    DEFAULT: ('⚠️ Были ли у вас раньше просрочки или дефолты по кредитам?',
              keyboard('default', DEFAULT_ON_FILE, DEFAULT_LABELS, per_row=2)),
    INTENT: ('🎯 Цель кредита?', keyboard('intent', LOAN_INTENT, INTENT_LABELS)),
    GRADE: ('📊 Кредитный грейд (A — лучший, G — худший):', keyboard('grade', LOAN_GRADE, GRADE_LABELS, per_row=2)),
    LOAN_AMOUNT: ('💰 Какую сумму кредита вы хотите получить (в долларах)?', None),
    LOAN_RATE: ('📈 Процентная ставка по кредиту, % (например, 8.5):', None),
}

# state -> (feature, min, max, integer)
NUMBER_FIELDS = {
    AGE: ('person_age', 18, 100, True),
    INCOME: ('person_income', 1_000, 10_000_000, True),
    EMP_LENGTH: ('person_emp_length', 0, 60, False),
    CRED_HIST: ('cb_person_cred_hist_length', 0, 60, True),
    LOAN_AMOUNT: ('loan_amnt', 100, 1_000_000, True),
    LOAN_RATE: ('loan_int_rate', 0, 50, False),
}

# state -> (feature, callback prefix, labels)
CHOICE_FIELDS = {
    HOME: ('person_home_ownership', 'home', HOME_LABELS),
    DEFAULT: ('cb_person_default_on_file', 'default', DEFAULT_LABELS),
    INTENT: ('loan_intent', 'intent', INTENT_LABELS),
    GRADE: ('loan_grade', 'grade', GRADE_LABELS),
}

RESTART_KEYBOARD = InlineKeyboardMarkup([[
    InlineKeyboardButton('🔄 Новая заявка', callback_data='restart'),
    InlineKeyboardButton('👋 Выход', callback_data='exit'),
]])

THOUSANDS = re.compile(r'^\d{1,3}(,\d{3})+(\.\d+)?$')


def parse_number(text: str) -> float | None:
    """Accepts '50000', '50 000', '50,000', '$50000', '8,5', '8.5%'."""
    s = re.sub(r'[\s_]', '', text).lstrip('$').rstrip('%')
    s = s.replace(',', '') if THOUSANDS.match(s) else s.replace(',', '.')
    try:
        value = float(s)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def money(value: float) -> str:
    return f'${value:,.0f}'.replace(',', ' ')


async def ask(message: Message, state: int) -> int:
    text, markup = PROMPTS[state]
    await message.reply_text(text, reply_markup=markup)
    return state


async def advance(message: Message, context: ContextTypes.DEFAULT_TYPE, state: int) -> int:
    idx = ORDER.index(state)
    if idx + 1 < len(ORDER):
        return await ask(message, ORDER[idx + 1])
    return await send_prediction(message, context)


def number_handler(state: int):
    key, lo, hi, integer = NUMBER_FIELDS[state]

    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        value = parse_number(update.message.text)
        if value is None or not lo <= value <= hi:
            await update.message.reply_text(f'❌ Введите число от {lo:,} до {hi:,}:'.replace(',', ' '))
            return state
        context.user_data['answers'][key] = int(round(value)) if integer else value
        return await advance(update.message, context, state)

    return handler


def choice_handler(state: int):
    key, _, labels = CHOICE_FIELDS[state]

    async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        query = update.callback_query
        await query.answer()
        value = query.data.split(':', 1)[1]
        context.user_data['answers'][key] = value
        await query.edit_message_text(f'{PROMPTS[state][0]}\n✅ {labels[value]}')
        return await advance(query.message, context, state)

    return handler


def format_result(answers: dict, risk: float, approved: bool, threshold: float) -> str:
    verdict = ('✅ КРЕДИТ, СКОРЕЕ ВСЕГО, ОДОБРЯТ' if approved
               else '❌ ВЫСОКИЙ РИСК — КРЕДИТ, СКОРЕЕ ВСЕГО, НЕ ОДОБРЯТ')
    a = answers
    return f"""{verdict}

📊 Оценка риска невозврата: {risk:.1%}
(заявка одобряется, если риск ниже {threshold:.0%})

📋 Ваши данные:
• Возраст: {a['person_age']}
• Годовой доход: {money(a['person_income'])}
• Жильё: {HOME_LABELS[a['person_home_ownership']]}
• Стаж на текущем месте, лет: {a['person_emp_length']:g}
• Кредитная история, лет: {a['cb_person_cred_hist_length']}
• Просрочки в прошлом: {DEFAULT_LABELS[a['cb_person_default_on_file']]}
• Цель: {INTENT_LABELS[a['loan_intent']]}
• Грейд: {a['loan_grade']}
• Сумма кредита: {money(a['loan_amnt'])}
• Ставка: {a['loan_int_rate']:g}%
• Сумма / доход: {a['loan_amnt'] / a['person_income']:.0%}

ℹ️ Это прогноз учебной модели, а не решение банка."""


async def send_prediction(message: Message, context: ContextTypes.DEFAULT_TYPE) -> int:
    model_data = context.bot_data['model']
    answers = context.user_data['answers']
    risk, approved = predict_risk(model_data, answers)
    logger.info('Prediction: risk=%.3f approved=%s', risk, approved)
    await message.reply_text(
        format_result(answers, risk, approved, model_data['threshold']), reply_markup=RESTART_KEYBOARD
    )
    return PREDICT


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['answers'] = {}
    await update.message.reply_text(
        '🏦 Добро пожаловать в бот прогноза одобрения кредита!\n\n'
        'Я задам 10 вопросов и оценю ваши шансы с помощью модели логистической регрессии.\n'
        'Отменить анкету можно командой /cancel.'
    )
    return await ask(update.message, AGE)


async def restart_or_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_reply_markup(reply_markup=None)
    if query.data == 'restart':
        context.user_data['answers'] = {}
        await query.message.reply_text('🔄 Начинаем новую заявку.')
        return await ask(query.message, AGE)
    await query.message.reply_text('👋 Спасибо, что воспользовались ботом! Чтобы начать заново — /start')
    return ConversationHandler.END


async def use_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('👆 Пожалуйста, выберите вариант кнопкой выше.')


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop('answers', None)
    await update.message.reply_text('❌ Анкета отменена. Чтобы начать заново — /start')
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        '📖 Помощь\n\n'
        'Бот оценивает риск невозврата кредита с помощью логистической регрессии, '
        'обученной на данных о кредитных заявках.\n\n'
        '🔄 Команды:\n'
        '/start — новая заявка\n'
        '/cancel — отменить текущую заявку\n'
        '/help — эта справка'
    )


def build_application(token: str, model_data: dict) -> Application:
    application = Application.builder().token(token).build()
    application.bot_data['model'] = model_data

    text = filters.TEXT & ~filters.COMMAND
    states = {s: [MessageHandler(text, number_handler(s))] for s in NUMBER_FIELDS}
    states |= {s: [CallbackQueryHandler(choice_handler(s), pattern=f'^{prefix}:')]
               for s, (_, prefix, _) in CHOICE_FIELDS.items()}
    states[PREDICT] = [CallbackQueryHandler(restart_or_exit, pattern='^(restart|exit)$')]

    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states=states,
        fallbacks=[CommandHandler('cancel', cancel), MessageHandler(text, use_buttons)],
        allow_reentry=True,
    ))
    application.add_handler(CommandHandler('help', help_command))
    return application


def main():
    load_dotenv()
    token = os.getenv('TELEGRAM_TOKEN')
    if not token or token.startswith('your_token_here'):
        logger.error('TELEGRAM_TOKEN не задан. Укажите его в .env (см. .env.example) или в переменной окружения.')
        return

    try:
        model_data = load_model()
    except FileNotFoundError:
        logger.error('Файл %s не найден. Сначала запустите: python train_model.py', MODEL_PATH)
        return

    application = build_application(token, model_data)
    logger.info('🚀 Бот запущен')
    application.run_polling()


if __name__ == '__main__':
    main()
