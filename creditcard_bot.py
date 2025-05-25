import json
from typing import List, Optional
from datetime import datetime, timedelta
import re
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

# Define tool schemas using Pydantic
class BillPaymentReminderSchema(BaseModel):
    card_name: str = Field(..., description="The name or identifier of the credit card for which the bill payment reminder is being set, e.g., 'ICICI Platinum', 'HDFC Millennia'.")
    due_date: Optional[str] = Field(None, description="The due date for the bill payment, e.g., '2025-07-15'. If not provided, defaults to 7 days from today.")

class CreditPointsCheckSchema(BaseModel):
    card_name: str = Field(..., description="The name or identifier of the credit card to check points for, e.g., 'Amex Gold'.")

class CardBalanceCheckSchema(BaseModel):
    card_name: str = Field(..., description="The name or identifier of the credit card to check balance for, e.g., 'SBI SimplySAVE'.")

class CreditCardBot:
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """Initialize the Credit Card Assistant Bot with LangChain."""
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            model=model_name,
        )
        self.model_name = model_name
        print(f"Initializing CreditCardBot with model: {self.model_name}")

        self.variables = {
            "editable": {
                "number_of_credit_cards": [1],
                "credit_card_names": ["HDFC Millennia"],
                "credit_card_limits": ["100000"], # Assuming limit is a string, can be float/int
                "alternate_email": ["neelabh.alt@example.com"]
            },
            "non_editable": {
                "name": "Neelabh",
                "primary_email": "neelabhverma@gmail.com",
                "phone_number": "8989898989"
            }
        }

        self.tools = [
            StructuredTool.from_function(
                func=self._set_bill_payment_reminder,
                name="set_bill_payment_reminder",
                description="Set a reminder for paying a credit card bill. Use this tool ONLY when the user explicitly requests to set a reminder for a specific card bill payment, e.g., 'remind me to pay my HDFC card bill' or 'set a payment reminder for Amex due on July 20th'.",
                args_schema=BillPaymentReminderSchema,
            ),
            StructuredTool.from_function(
                func=self._check_credit_points,
                name="check_credit_points",
                description="Check the accumulated credit or reward points for a specific credit card. Use this tool ONLY when the user explicitly asks about their points for a named card, e.g., 'how many points do I have on my ICICI card?'.",
                args_schema=CreditPointsCheckSchema,
            ),
            StructuredTool.from_function(
                func=self._check_card_balance,
                name="check_card_balance",
                description="Check the current outstanding balance and available credit limit for a specific credit card. Use this tool ONLY when the user explicitly asks about their card balance for a named card, e.g., 'what's the balance on my SBI card?'.",
                args_schema=CardBalanceCheckSchema,
            ),
        ]
        self.conversation_history: List[BaseMessage] = []

    def _get_current_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_system_prompt(self):
        user_profile = (
            f"- Name: {self.variables['non_editable']['name']}\n"
            f"- Primary Email: {self.variables['non_editable']['primary_email']}\n"
            f"- Phone Number: {self.variables['non_editable']['phone_number']}"
        )

        card_info = (
            f"- Number of Credit Cards: {', '.join(map(str, self.variables['editable']['number_of_credit_cards']))}\n"
            f"- Credit Card(s) Owned: {', '.join(self.variables['editable']['credit_card_names'])}\n"
            f"- Credit Card Limit(s): {', '.join(self.variables['editable']['credit_card_limits'])}\n"
            f"- Alternate Email: {', '.join(self.variables['editable']['alternate_email'])}"
        )

        return f"""
## Role and Identity
You are a helpful AI assistant specializing in credit card related queries.
Your goal is to assist users with their credit card questions, set bill payment reminders, check credit points, and check card balances.
You should be polite, professional, and accurate.
Current time: {self._get_current_time()}

## User Profile
{user_profile}

## User's Credit Card Information (Editable by user interaction)
{card_info}

## Core Responsibilities
1. Use the `set_bill_payment_reminder` tool ONLY when the user explicitly asks to set a reminder for a card bill payment, providing the card name. If due date is missing, default to 7 days from today.
2. Use the `check_credit_points` tool ONLY when the user explicitly asks about reward points for a specific card name.
3. Use the `check_card_balance` tool ONLY when the user explicitly asks about the balance for a specific card name.
4. Update user's credit card information when new details (e.g., new card, updated limit, alternate email) are shared.
5. Provide general advice or answer questions about credit cards, fees, benefits, etc., without using tools if the query doesn't match tool functionalities.

## Tool Usage Instructions
- Use tools ONLY for their specified explicit purpose. Do not guess or infer.
- `set_bill_payment_reminder`: Requires `card_name`. `due_date` is optional (defaults to 7 days from now).
- `check_credit_points`: Requires `card_name`.
- `check_card_balance`: Requires `card_name`.
- If required parameters for any tool are missing or unclear, ask for clarification instead of using the tool with incomplete data.
- Responses from tool calls will be appended to the dialogue for final response generation.

## Response Format Guidelines
- Keep responses concise and to the point.
- For bill reminders, confirm the card name and reminder date.
- For credit points, state the points and for which card.
- For balance checks, provide outstanding balance and available limit for the specified card.
- If information is missing or unclear for a tool, politely ask the user for the necessary details.
- Use a professional and friendly tone.
"""

    def _update_variables(self, user_input: str):
        """Extract and update editable variables from user input."""
        extraction_prompt = f"""
        Extract job-related information from this user message:
        "{user_input}"

        Current editable variables: {self.variables['editable']}

        INSTRUCTIONS:
        1. Analyze for mentions of user wanting to change or add new details to Number of Credit Cards, Credit Card(s) Owned, Credit Card Limit(s), Alternate Email.
        2. Return ONLY a JSON object with fields to update, adding new values to lists. Do not include any explanations or additional text.
        3. If no relevant information is found, return an empty JSON object {{}}.
        """
        
        # print(f"[Variable Extraction] Using model: {self.model_name}")
        try:
            # Use LangChain's invoke method for variable extraction
            response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
            content = response.content.strip()
            # print(f"[Variable Extraction debug message] Model response: {content}")
            
            if content:
                updates = json.loads(content)
                # print(f"[Variable Extraction] Detected updates: {updates}")
                
                for key, value in updates.items():
                    if key in self.variables["editable"]:
                        if isinstance(value, list):
                            for item in value:
                                if item not in self.variables["editable"][key]:
                                    self.variables["editable"][key].append(item)
                        elif value:
                            if value not in self.variables["editable"][key]:
                                self.variables["editable"][key].append(value)
            else:
                print("[Variable Extraction] No updates detected")
        except Exception as e:
            print(f"[Variable Extraction] Error: {e}")


    def _set_bill_payment_reminder(self, card_name: str, due_date: Optional[str] = None) -> str:
        """Sets a reminder for credit card bill payment."""
        try:
            if not card_name or not card_name.strip():
                return "Error: Credit card name cannot be empty for setting a reminder."

            if not due_date:
                due_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            else:
                # Validate date format
                try:
                    datetime.strptime(due_date, '%Y-%m-%d')
                except ValueError:
                    return "Error: Invalid due date format. Please use YYYY-MM-DD."

            return f"Reminder set: Pay bill for {card_name} by {due_date}."
        except Exception as e:
            print(f"[Bill Reminder Error] {e}")
            return f"Error setting bill payment reminder: {str(e)}"

    def _check_credit_points(self, card_name: str) -> str:
        """Checks (dummy) credit points for a given card."""
        try:
            if not card_name or not card_name.strip():
                return "Error: Credit card name cannot be empty for checking points."
            # In a real scenario, this would query a database or API
            # For now, provide a dummy response
            points = "5,000" # Example points
            if "HDFC" in card_name:
                points = "7,250"
            elif "Amex" in card_name:
                points = "12,800"
            return f"You have {points} reward points on your {card_name} card."
        except Exception as e:
            print(f"[Credit Points Error] {e}")
            return f"Error checking credit points: {str(e)}"

    def _check_card_balance(self, card_name: str) -> str:
        """Checks (dummy) balance for a given card."""
        try:
            if not card_name or not card_name.strip():
                return "Error: Credit card name cannot be empty for checking balance."
            # In a real scenario, this would query a database or API
            # For now, provide a dummy response
            outstanding_balance = "₹15,230.50"
            available_limit = "₹84,769.50"
            if "ICICI" in card_name:
                outstanding_balance = "₹5,100.00"
                available_limit = "₹1,94,900.00"
            elif "SBI" in card_name:
                outstanding_balance = "₹22,000.75"
                available_limit = "₹1,27,999.25"

            return (f"For your {card_name} card: "
                    f"Outstanding balance is {outstanding_balance}. "
                    f"Available credit limit is {available_limit}.")
        except Exception as e:
            print(f"[Card Balance Error] {e}")
            return f"Error checking card balance: {str(e)}"

    def chat(self, user_input: str):
        """Process user input with LangChain tool calling."""
        print(f"\n[Query Processing] '{user_input}'")

        self._update_variables(user_input)
        self.conversation_history.append(HumanMessage(content=user_input))

        try:
            messages = [
                SystemMessage(content=self._build_system_prompt()),
                *self.conversation_history[-6:], # Keep context window manageable
            ]

            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)

            if response.tool_calls:
                self.conversation_history.append(response)
                tool_messages_for_final_response = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(f"[Tool Call] Function: {tool_name}, Arguments: {tool_args}")
                    
                    # Find and execute the tool
                    tool_executed = False
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                result = tool.invoke(tool_args)
                                tool_executed = True
                            except Exception as e:
                                print(f"Error invoking tool {tool_name} with args {tool_args}: {e}")
                                result = f"Error executing tool {tool_name}: {e}"
                            
                            tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                            self.conversation_history.append(tool_message) # Add to full history
                            tool_messages_for_final_response.append(tool_message) # Add to current turn context
                            break
                    if not tool_executed:
                        print(f"Tool {tool_name} not found or failed to execute.")
                        # Add a message indicating failure if necessary
                        error_tool_msg = ToolMessage(content=f"Tool {tool_name} could not be executed.", tool_call_id=tool_call["id"])
                        self.conversation_history.append(error_tool_msg)
                        tool_messages_for_final_response.append(error_tool_msg)

                # Generate final response after tool execution(s)
                final_messages_for_llm = [
                    SystemMessage(content=self._build_system_prompt()),
                    *self.conversation_history[-6:] # Use updated history
                ]
                
                final_response = self.llm.invoke(final_messages_for_llm)
                self.conversation_history.append(AIMessage(content=final_response.content))
                return final_response.content
            else:
                self.conversation_history.append(AIMessage(content=response.content))
                return response.content

        except Exception as e:
            print(f"[Error in chat] {e}")
            # Log the full error traceback for debugging
            import traceback
            traceback.print_exc()
            return "I'm experiencing technical difficulties. Please try again."

def test_credit_card_bot():
    TOGETHER_API_KEY = "0e2df1900707fffbc6c0a7254e6f091e707f69aa68094bb1fa4f7b6c1d484bcb" # Replace with your actual key


    bot = CreditCardBot(
        api_key=TOGETHER_API_KEY,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        # model_name="meta-llama/Llama-3.1-8B-Instruct-Turbo" # Smaller model for faster testing if needed
    )

    queries = [
        "What is the outstanding amount for my HDFC card and also set a reminder for its payment by next Friday and also tell me my points.",
        "Hi, can you tell me about credit card annual fees?", # General query
        "Set a reminder to pay my HDFC Millennia card bill.", # Tool: set_bill_payment_reminder (no date)
        "What's the balance on my SBI SimplySAVE card?", # Tool: check_card_balance
        "What's the balance on my HDFC Millennia card? and also check my points.", # Tool: check_card_balance and check_credit_points
        "How many points do I have on my Amex Gold?", # Tool: check_credit_points
        "Remind me to pay my ICICI Platinum bill on 2025-08-10.", # Tool: set_bill_payment_reminder (with date)
        "I just got a new card, it's an Axis Magnus. Its limit is 5 lakhs.", # Variable update
        "What are my current card details?", # General query, should summarize from variables
        "My alternate email is now new.alt@example.com.", # Variable update
        "Can you check points for my Visa card?", # Tool: check_credit_points (generic, might use default or ask for more specific name)
        "Set a reminder for my shopping expenses.", # Ambiguous, should not trigger payment reminder
        "I have an HDFC card, an ICICI card, and an Amex. Update my card count.", # Variable update (complex - LLM might struggle to update count perfectly without specific instruction)
        "My credit card names are HDFC Infinia and SBI Elite. My limit for HDFC is 2 lakhs and for SBI is 1.5 lakhs." # Variable update
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = bot.chat(query)
        print("###################################################################################")
        print(f"Bot: {response}")
        print(f"Conversation History Length: {len(bot.conversation_history)}")
        print("-"*80)
print(f"\nUpdated Editable Variables: {json.dumps(bot.variables['editable'], indent=2)}")
if __name__ == "__main__":
    test_credit_card_bot()
