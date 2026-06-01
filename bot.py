"""
Telegram bot for credit approval prediction
Uses trained logistic regression model
"""
import logging
import pickle
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
from telegram.error import TelegramError

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Dialog states
AGE, INCOME, HOME, EMP_LENGTH, INTENT, GRADE, LOAN_AMOUNT, LOAN_RATE, PREDICT = range(9)

# Dictionary to store user data
user_data = {}

# Dictionaries for encoding categorical variables
HOMES = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
INTENTS = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOME_IMPROVEMENT': 4, 'DEBT_CONSOLIDATION': 5}
GRADES = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

def load_model():
    """Loads the trained model"""
    try:
        with open('credit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        logger.error("File credit_model.pkl not found. Please run train_model.py first")
        return None

# Load model on startup
MODEL_DATA = load_model()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start bot and begin dialog"""
    user_id = update.effective_user.id
    user_data[user_id] = {}
    
    message = """🏦 Welcome to Credit Approval Prediction Bot!

I'll help you find out the probability of getting a loan based on logistic regression.

Let's get started! 📋

How old are you?"""
    
    await update.message.reply_text(message)
    return AGE

async def get_age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get age"""
    user_id = update.effective_user.id
    try:
        age = int(update.message.text)
        if age < 18 or age > 120:
            await update.message.reply_text("⚠️ Age must be between 18 and 120 years old. Try again:")
            return AGE
        user_data[user_id]['age'] = age
        
        await update.message.reply_text("✅ Thank you! Now enter your annual income (in dollars):")
        return INCOME
    except ValueError:
        await update.message.reply_text("❌ Please enter a number:")
        return AGE

async def get_income(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get income"""
    user_id = update.effective_user.id
    try:
        income = int(update.message.text)
        if income < 0:
            await update.message.reply_text("⚠️ Income cannot be negative. Try again:")
            return INCOME
        user_data[user_id]['income'] = income
        
        # Create buttons for housing type selection
        keyboard = [
            [InlineKeyboardButton("Rent (RENT)", callback_data='home_RENT')],
            [InlineKeyboardButton("Own (OWN)", callback_data='home_OWN')],
            [InlineKeyboardButton("Mortgage (MORTGAGE)", callback_data='home_MORTGAGE')],
            [InlineKeyboardButton("Other (OTHER)", callback_data='home_OTHER')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text("🏠 What is your housing type?", reply_markup=reply_markup)
        return HOME
    except ValueError:
        await update.message.reply_text("❌ Please enter a number:")
        return INCOME

async def get_home(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get housing type"""
    user_id = update.effective_user.id
    query = update.callback_query
    await query.answer()
    
    home_type = query.data.split('_')[1]
    user_data[user_id]['home'] = HOMES[home_type]
    
    await query.edit_message_text("✅ Thank you!\n\nHow many years have you been working? (enter a number, can be 0):")
    return EMP_LENGTH

async def get_emp_length(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get employment length"""
    user_id = update.effective_user.id
    try:
        emp_length = float(update.message.text)
        if emp_length < 0 or emp_length > 100:
            await update.message.reply_text("⚠️ Employment length must be between 0 and 100 years. Try again:")
            return EMP_LENGTH
        user_data[user_id]['emp_length'] = emp_length
        
        # Buttons for loan purpose selection
        keyboard = [
            [InlineKeyboardButton("Personal (PERSONAL)", callback_data='intent_PERSONAL')],
            [InlineKeyboardButton("Education (EDUCATION)", callback_data='intent_EDUCATION')],
            [InlineKeyboardButton("Medical (MEDICAL)", callback_data='intent_MEDICAL')],
            [InlineKeyboardButton("Venture (VENTURE)", callback_data='intent_VENTURE')],
            [InlineKeyboardButton("Home Improvement (HOME_IMPROVEMENT)", callback_data='intent_HOME_IMPROVEMENT')],
            [InlineKeyboardButton("Debt Consolidation (DEBT_CONSOLIDATION)", callback_data='intent_DEBT_CONSOLIDATION')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text("💡 What is the purpose of your loan?", reply_markup=reply_markup)
        return INTENT
    except ValueError:
        await update.message.reply_text("❌ Please enter a number:")
        return EMP_LENGTH

async def get_intent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get loan purpose"""
    user_id = update.effective_user.id
    query = update.callback_query
    await query.answer()
    
    intent = query.data.split('_', 1)[1]
    user_data[user_id]['intent'] = INTENTS[intent]
    
    # Buttons for loan grade selection
    keyboard = [
        [InlineKeyboardButton("A (Low Risk)", callback_data='grade_A'), InlineKeyboardButton("B", callback_data='grade_B')],
        [InlineKeyboardButton("C", callback_data='grade_C'), InlineKeyboardButton("D", callback_data='grade_D')],
        [InlineKeyboardButton("E", callback_data='grade_E'), InlineKeyboardButton("F", callback_data='grade_F')],
        [InlineKeyboardButton("G (High Risk)", callback_data='grade_G')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text("📈 Select loan grade (A - best, G - worst):", reply_markup=reply_markup)
    return GRADE

async def get_grade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get loan grade"""
    user_id = update.effective_user.id
    query = update.callback_query
    await query.answer()
    
    grade = query.data.split('_')[1]
    user_data[user_id]['grade'] = GRADES[grade]
    
    await query.edit_message_text("💰 What loan amount are you requesting? (in dollars)")
    return LOAN_AMOUNT

async def get_loan_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get loan amount"""
    user_id = update.effective_user.id
    try:
        loan_amount = int(update.message.text)
        if loan_amount <= 0 or loan_amount > 1000000:
            await update.message.reply_text("⚠️ Amount must be between 1 and 1,000,000 dollars. Try again:")
            return LOAN_AMOUNT
        user_data[user_id]['loan_amount'] = loan_amount
        
        await update.message.reply_text("📈 Enter interest rate (example: 8.5):")
        return LOAN_RATE
    except ValueError:
        await update.message.reply_text("❌ Please enter a number:")
        return LOAN_AMOUNT

async def get_loan_rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем процентную ставку и делаем прогноз"""
    user_id = update.effective_user.id
    try:
        loan_rate = float(update.message.text)
        if loan_rate < 0 or loan_rate > 50:
            await update.message.reply_text("⚠️ Ставка должна быть от 0 до 50%. Попробуйте ещё раз:")
            return LOAN_RATE
        user_data[user_id]['loan_rate'] = loan_rate
        
        # Calculate loan_percent_income
        income = user_data[user_id]['income']
        loan_amount = user_data[user_id]['loan_amount']
        loan_percent_income = loan_amount / income if income > 0 else 0
        
        # Prepare data for model
        user_features = {
            'person_age': user_data[user_id]['age'],
            'person_income': user_data[user_id]['income'],
            'person_home_ownership': user_data[user_id]['home'],
            'person_emp_length': user_data[user_id]['emp_length'],
            'loan_intent': user_data[user_id]['intent'],
            'loan_grade': user_data[user_id]['grade'],
            'loan_amnt': user_data[user_id]['loan_amount'],
            'loan_int_rate': user_data[user_id]['loan_rate'],
            'loan_percent_income': loan_percent_income,
            'cb_person_default_on_file': 0,  # Assume no defaults
            'cb_person_cred_hist_length': 5  # Average value
        }
        
        # Make prediction
        if MODEL_DATA is None:
            await update.message.reply_text("❌ Error: Model not loaded. Please contact administrator.")
            return ConversationHandler.END
        
        model = MODEL_DATA['model']
        scaler = MODEL_DATA['scaler']
        feature_names = MODEL_DATA['feature_names']
        
        # Create DataFrame with correct feature order
        X = pd.DataFrame([user_features])[feature_names]
        X_scaled = scaler.transform(X)
        
        # Get prediction and probability
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        # Format response
        if prediction == 1:
            result = "✅ CREDIT APPROVED!"
            probability_pct = probability[1] * 100
            message = f"""{result}

📊 Prediction Result:
Approval Probability: {probability_pct:.1f}%

📋 Your Parameters:
• Age: {user_data[user_id]['age']} years
• Income: ${user_data[user_id]['income']:,}
• Loan Amount: ${user_data[user_id]['loan_amount']:,}
• Interest Rate: {user_data[user_id]['loan_rate']}%
• Loan-to-Income Ratio: {loan_percent_income:.2%}

✨ Congratulations! Based on the model analysis, your application has strong chances of approval."""
        else:
            result = "❌ CREDIT REJECTED"
            probability_pct = probability[0] * 100
            message = f"""{result}

📊 Prediction Result:
Rejection Probability: {probability_pct:.1f}%

📋 Your Parameters:
• Age: {user_data[user_id]['age']} years
• Income: ${user_data[user_id]['income']:,}
• Loan Amount: ${user_data[user_id]['loan_amount']:,}
• Interest Rate: {user_data[user_id]['loan_rate']}%
• Loan-to-Income Ratio: {loan_percent_income:.2%}

⚠️ Unfortunately, according to the model prediction, your application does not meet the approval criteria.
We recommend contacting the bank branch for personal consultation."""
        
        await update.message.reply_text(message)
        
        # Offer to start over
        keyboard = [
            [InlineKeyboardButton("New Application", callback_data='restart')],
            [InlineKeyboardButton("Exit", callback_data='exit')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("\nWhat would you like to do?", reply_markup=reply_markup)
        
        return PREDICT
        
    except ValueError:
        await update.message.reply_text("❌ Please enter a number:")
        return LOAN_RATE

async def restart_or_exit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restart or exit"""
    user_id = update.effective_user.id
    query = update.callback_query
    await query.answer()
    
    if query.data == 'restart':
        user_data[user_id] = {}
        await query.edit_message_text("🔄 Starting new application...\\n\\nHow old are you?")
        return AGE
    else:
        await query.edit_message_text("👋 Thank you for using the bot! Goodbye!")
        return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel dialog"""
    await update.message.reply_text("❌ Dialog cancelled. Enter /start to begin.")
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help"""
    help_text = """📖 Bot Help

This bot helps predict the probability of loan approval based on logistic regression.

🔄 Commands:
/start - Start a new application
/help - Show this help
/cancel - Cancel current application

📝 Process:
1. Enter your information (age, income, housing type, etc.)
2. The bot will process the information using a machine learning model
3. Get the prediction result with approval probability

⚙️ The model is trained on real credit approval data."""
    
    await update.message.reply_text(help_text)

def main():
    """Main function to run the bot"""
    # Get token from environment variable or .env file
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    TOKEN = os.getenv('TELEGRAM_TOKEN')
    
    if not TOKEN or TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("\n⚠️ ERROR: TELEGRAM_TOKEN not set!")
        print("Please set your token in the .env file or as an environment variable.")
        print("See .env.example for details.")
        logger.error("TELEGRAM_TOKEN not configured")
        return
    
    # Create application
    application = Application.builder().token(TOKEN).build()
    
    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_age)],
            INCOME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_income)],
            HOME: [CallbackQueryHandler(get_home, pattern='^home_')],
            EMP_LENGTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_emp_length)],
            INTENT: [CallbackQueryHandler(get_intent, pattern='^intent_')],
            GRADE: [CallbackQueryHandler(get_grade, pattern='^grade_')],
            LOAN_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_loan_amount)],
            LOAN_RATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_loan_rate)],
            PREDICT: [CallbackQueryHandler(restart_or_exit, pattern='^(restart|exit)')],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True
    )
    
    # Add handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('start', start))
    
    # Run the bot
    print("🚀 Bot started!")
    application.run_polling()

if __name__ == '__main__':
    main()
