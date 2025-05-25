import json
from typing import List, Optional
from serpapi import GoogleSearch
from datetime import datetime, timedelta
import re
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field


class JobSearchSchema(BaseModel):
    job_title: str = Field(..., description="The specific job role eg. 'Python Developer','ML Engineer'.")
    job_location: str = Field(..., description="The specific job location eg. 'New York', 'Remote'.")

class ScheduleMeetingSchema(BaseModel):
    user_email: str = Field(..., description="The user's email address for the meeting.")
    recipient_email: str = Field(..., description="The email address of the person the meeting is with.")
    date: str = Field(..., description="The date of the meeting, e.g., '2025-06-01'.")
    time: str = Field(..., description="The time of the meeting, e.g., '14:00'.")

class FollowUpReminderSchema(BaseModel):
    company_name: str = Field(..., description="The name of the company to follow up with.")
    date: Optional[str] = Field(None, description="The date for the follow-up reminder, e.g., '2025-06-01'. If not provided, defaults to 3 days from today.")

class JobFindingBot:
    def __init__(self, api_key: str, serpapi_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """Initialize the Job Finding Assistant Bot with LangChain."""
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            model=model_name,
        )
        self.serpapi_key = serpapi_key
        self.model_name = model_name
        print(f"Initializing JobFindingBot with model: {self.model_name}")

        
        self.variables = {
            "editable": {
                "notice_period": ["20 days"],
                "expected_salary": ["80000"],
                "job_location_preferred": ["Remote", "New York"],
                "email": ["neelabhverma@gmail.com"],
                "number": ["8989898989"],
                "Interested_Roles": ["software engineer", "data analyst"]
            },
            "non_editable": {
                "name": "Neelabh",
                "degree_holding": "BTech in Computer Science"
            }
        }

        
        self.tools = [
            StructuredTool.from_function(
                func=self._search_jobs,
                name="search_jobs",
                description="Search for job listings only when the user explicitly requests to find or show job opportunities, such as 'find software engineer jobs in Seattle' or 'show data analyst positions'. Do not use for any other use case other than the Job search.",
                args_schema=JobSearchSchema,
            ),
            StructuredTool.from_function(
                func=self._schedule_meeting,
                name="schedule_meeting",
                description="Schedule a meeting only when the user explicitly requests to arrange or set up a meeting, providing their email, the recipient's email, date, and time, such as 'schedule a meeting with hr@company.com on 2025-06-01 at 14:00'. Do not use for other queries or if any required details are missing.",
                args_schema=ScheduleMeetingSchema,
            ),
            StructuredTool.from_function(
                func=self._follow_up_reminder,
                name="follow_up_reminder",
                description="Set a reminder for following up on a job application only when the user explicitly requests to set or arrange a follow-up reminder for a company they applied to, such as 'set a follow-up reminder for Apple'. If no date is specified, use a date 3 days from today. Do not use for other queries",
                args_schema=FollowUpReminderSchema,
            ),
        ]

        self.conversation_history: List[BaseMessage] = []

    def _get_current_time(self):
        """Return the current time as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_system_prompt(self):
        """Build the system prompt with user profile and preferences."""
        user_profile = (
            f"- Name: {self.variables['non_editable']['name']}\n"
            f"- Education: {self.variables['non_editable']['degree_holding']}"
        )
        
        job_preferences = (
            f"- Notice Period: {', '.join(self.variables['editable']['notice_period'])}\n"
            f"- Expected Salary: {', '.join(str(s) for s in self.variables['editable']['expected_salary'])}\n"
            f"- Preferred Locations: {', '.join(self.variables['editable']['job_location_preferred'])}\n"
            f"- Contact Email: {', '.join(self.variables['editable']['email'])}\n"
            f"- Contact Number: {', '.join(self.variables['editable']['number'])}\n"
            f"- Interested Roles: {', '.join(self.variables['editable']['Interested_Roles'])}"
        )

        return f"""
## Role and Identity
You are a professional job finding assistant with expertise in career advice, job search, meeting scheduling, and follow-up reminders.
Your goal is to help users find job opportunities, schedule meetings, set follow-up reminders, or provide career advice based on their preferences.
Current time: {self._get_current_time()}

## User Profile
{user_profile}

## Job Preferences
{job_preferences}

## Core Responsibilities
1. Use the search_jobs tool only when the user explicitly requests to find or show job listings with a clear job title and location.
2. Use the schedule_meeting tool only when the user explicitly requests to arrange or set up a meeting with clear email, date, and time details.
3. Use the follow_up_reminder tool only when the user explicitly requests to set or arrange a follow-up reminder for a job application with a clear company name.
4. Update user information when new details (e.g., experience, salary, location preferences) are shared.
5. Provide career advice, skill recommendations, or summaries of user information for general queries without using tools.

## Tool Usage Instructions
- Use the search_jobs tool ONLY for explicit job search requests, such as 'find software engineer jobs in Seattle' or 'show data analyst positions in Chicago'. Do NOT use for general career advice, skill questions, or if job title or location is missing or unclear use the user interested roles and prefered location varialbe to do the search .
- Use the schedule_meeting tool ONLY for explicit meeting scheduling requests, such as 'schedule a meeting with hr@company.com on 2025-06-01 at 14:00'. Do NOT use for other queries or if user email, recipient email, date, or time is missing.
- Use the follow_up_reminder tool ONLY for explicit follow-up reminder requests, such as 'set a follow-up reminder for Apple' or 'set a follow-up reminder for Apple on 2025-06-01'. If no date is provided, use a date 3 days from today. Do NOT use for other queries or if company name is missing.
- If required parameters for any tool are missing or unclear, respond with a helpful message asking for clarification instead of using the tool.
- Responses from tool calls will be appended to the dialogue for final response generation.

## Response Format Guidelines
- Keep responses concise, under 3 sentences when possible.
- For job search results,Always include the include job titles, companies, locations, salaries, and apply links and also just summaries the descripton of the job as these ae the bare Minimum reqirements for the.
- If no jobs are found, inform the user politely.
- For meeting scheduling, confirm the meeting details with emails, date, and time.
- For follow-up reminders, confirm the reminder details with company name and date.
- Use a professional, friendly tone and align responses with user preferences.
- If information is missing or unclear, admit it and ask for clarification.
"""

    def _update_variables(self, user_input: str):
        """Extract and update editable variables from user input."""
        extraction_prompt = f"""
        Extract job-related information from this user message:
        "{user_input}"

        Current editable variables: {self.variables['editable']}

        INSTRUCTIONS:
        1. Analyze for mentions of user wanting to change or add new details to notice period, salary, job locations, contact info (email or phone), or interested roles.
        2. Return ONLY a JSON object with fields to update, adding new values to lists. Do not include any explanations or additional text.
        3. If no relevant information is found, return an empty JSON object {{}}.
        """
        
        # print(f"[Variable Extraction] Using model: {self.model_name}")
        try:
            
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

    def _search_jobs(self, job_title: str, job_location: str) -> str:
        """Search for jobs using SerpAPI with improved formatting."""
        try:
            params = {
                "engine": "google_jobs",
                "q": job_title,
                "location": job_location,
                "hl": "en",
                "api_key": self.serpapi_key
            }
            
            print(f"[SerpAPI] Searching for: {job_title} in {job_location}")
            search = GoogleSearch(params)
            results = search.get_dict()
            jobs_results = results.get("jobs_results", [])
            
            if not jobs_results:
                return f"No jobs found for {job_title} in {job_location}."

            response_parts = []
            for i, job in enumerate(jobs_results[:3], 1):
                title = job.get("title", "Unknown position")
                company = job.get("company_name", "Unknown company")
                location = job.get("location", "Unknown location")
                salary = job.get("salary", "$80,000 - $110,000 per year")
                
                description = job.get('description', 'No description provided.')
                if len(description) > 300:
                    description = description[:297] + "..."
                
                job_info = f"Job #{i}: {title} at {company} ({location})\n"
                job_info += f"Brief Description: {description}\n"
                job_info += f"Salary: {salary}\n"
                
                apply_links = job.get("apply_options", [])
                if apply_links and len(apply_links) > 0:
                    job_info += f"Apply Link: {apply_links[0].get('link', 'No link provided.')}\n"
                
                response_parts.append(job_info)
            
            return "\n\n".join(response_parts)
        except Exception as e:
            print(f"[SerpAPI Error] {e}")
            return f"Error searching for jobs: {str(e)}"

    def _schedule_meeting(self, user_email: str, recipient_email: str, date: str, time: str) -> str:
        """Dummy meeting scheduler that confirms meeting details."""
        try:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, user_email) or not re.match(email_pattern, recipient_email):
                return "Error: Invalid email format provided."

            try:
                datetime.strptime(date, '%Y-%m-%d')
                datetime.strptime(time, '%H:%M')
            except ValueError:
                return "Error: Invalid date (YYYY-MM-DD) or time (HH:MM) format."

            return f"Meeting scheduled between {user_email} and {recipient_email} on {date} at {time}."
        except Exception as e:
            print(f"[Meeting Scheduler Error] {e}")
            return f"Error scheduling meeting: {str(e)}"

    def _follow_up_reminder(self, company_name: str, date: Optional[str] = None) -> str:
        """Dummy follow-up reminder that confirms reminder details."""
        try:
            if not company_name or not company_name.strip():
                return "Error: Company name cannot be empty."

            if not date:
                date = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')

            try:
                datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                return "Error: Invalid date format (use YYYY-MM-DD)."

            return f"Follow-up reminder set for {company_name} on {date}."
        except Exception as e:
            print(f"[Follow-Up Reminder Error] {e}")
            return f"Error setting follow-up reminder: {str(e)}"

    def chat(self, user_input: str):
        """Process user input with LangChain tool calling."""
        print(f"\n[Query Processing] '{user_input}'")
        
        self._update_variables(user_input)
        self.conversation_history.append(HumanMessage(content=user_input))
        
        try:
            
            messages = [
                SystemMessage(content=self._build_system_prompt()),
                *self.conversation_history[-6:],
            ]
            
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)
            
            if response.tool_calls:
                self.conversation_history.append(response)  
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(f"[Tool Call] Function: {tool_name}, Arguments: {tool_args}")
                    for tool in self.tools:
                        if tool.name == tool_name:
                            result = tool.invoke(tool_args)
                            tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
                            # print(f"[Tool Response] {tool_name} result: {result}")
                            self.conversation_history.append(tool_message)
                            break
                
                
                final_messages = [
                    SystemMessage(content=self._build_system_prompt()),
                    *self.conversation_history[-6:],
                ]
                final_response = self.llm.invoke(final_messages)
                self.conversation_history.append(AIMessage(content=final_response.content))
                return final_response.content
            else:
                
                self.conversation_history.append(AIMessage(content=response.content))
                return response.content
                
        except Exception as e:
            print(f"[Error in chat] {e}")
            return "I'm experiencing technical difficulties. Please try again."
 
def test_job_bot():
    TOGETHER_API_KEY = "YOUR_TOGETHER_API_KEY_HERE"  
    SERPAPI_API_KEY = "YOUR_SERPAPI_API_KEY_HERE"  
    
    # Change the model name as needed
    bot = JobFindingBot(
        api_key=TOGETHER_API_KEY, 
        serpapi_key=SERPAPI_API_KEY,
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        # model_name="meta-llama/Llama-3.2-3B-Instruct-Turbo"
    )
    
    queries = [
        #Your Query Goes here try -- "Search DevOps jobs in Noida and set a follow-up reminder for Apple and also set a meeting with harsh his mail is harsh@gmail.com.",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = bot.chat(query)
        print("###################################################################################")
        print(f"Bot: {response}")
        print("-"*50)
        print(f"\nUpdated Editable Variables: {json.dumps(bot.variables['editable'], indent=2)}")
if __name__ == "__main__":
    test_job_bot()