from openai import OpenAI
import json
from datetime import datetime
import cml.data_v1 as cmldata

# db stuff
CONNECTION_NAME = "default-impala-aws"
conn = cmldata.get_connection(CONNECTION_NAME)

# openai stuff
client = OpenAI()


def retrieveCustomerInfo(name, dob, address):

    # name = 'Alex'
    # dob = '1990-03-13'
    # address = 'Sample Street 1'

    SQL_QUERY = f"""SELECT name,
    customer_id as Customer_ID,
    current_product as Current_Product,
    churn_risk as Churn_Risk,
    FROM_TIMESTAMP(customer_since, 'yyyy-MM-dd') as Customer_Since,
    FROM_TIMESTAMP(date_of_birth, 'yyyy-MM-dd') as Date_of_birth,
    address as Address,
    preapproved_for_discount as Preappoved_for_discount
    FROM afrank_SKO_demo
    WHERE LOWER(name) = LOWER('{name}')
    AND date_of_birth = '{dob}'
    AND LOWER(address) = LOWER('{address}');"""

    db_conn = conn.get_base_connection()

    db_cursor = conn.get_cursor()
    db_cursor.execute(SQL_QUERY)
    # Fetch the column names from the cursor
    columns = [desc[0] for desc in db_cursor.description]

    db_data = []

    for row in db_cursor:
        # Create a dictionary with column names as keys and row values as values
        row_data = dict(zip(columns, row))
        db_data.append(row_data)

    if len(db_data) == 0:
        return False
    else:
        return db_data[0]


def is_valid_date(date_str):
    try:
        # Attempt to parse the string into a datetime object
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        # If parsing fails, it's not a valid date
        return False


def predict(data: dict[str, str]) -> dict:
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    if "text" not in data:
        raise TypeError("data must contain a key of 'text'")

    if "task" not in data:
        raise TypeError("data must contain a key of 'task'")

    task = data['task']
    if not isinstance(task, str):
        raise TypeError("text must be a string")

    text = data["text"]
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if task == 'ai_help':
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are helping a call center worker for a telco company called airwave. You will receive the user text content, which will include the call center worker and the customers spoken words converted to text. You are called upon when teh call center agent needs some sort of help, like details about the available products or suggestions for troubleshooting etc. It is your job to provide helpful suggestions. Make sure they are short so the call center worker can easily look at them and read them out to the customer. Do not include multiple suggestions, just one with enough information that the call center agent can use. If the call center agend mentions they are looking for some information, or the customer sais they need additional information, provide that information (if available) in your response. You may receive additional information about the customer such as name and currently used products. The company has currently has 3 products: " + """
                AirSpeed Advanced
                High Speed Wireless Broadband

                Special Offer € 45 per month
                Cost is € 45 p/m for the first 3 months and € 60 thereafter
                12 Month contract
                Up to 70 Mbps / 7 Mbps

                FREE Fritzbox Router
                Installation fee of € 150
                Optional Home Phone Service

                and

                AirSpeed Plus
                High Speed Wireless Broadband

                Special Offer € 35 per month
                Cost is € 35 p/m for the first 3 months and € 50 thereafter
                12 Month contract
                Up to 50 Mbps / 5 Mbps

                FREE Fritzbox Router
                Installation fee of € 150
                Optional Home Phone Service

                and

                AirSpeed Home
                High Speed Wireless Broadband

                Special Offer € 25 per month
                Cost is € 25 p/m for the first 3 months and € 40 thereafter
                12 Month contract
                Up to 30 Mbps / 3 Mbps

                FREE Fritzbox Router
                Installation fee of €150
                Optional Home Phone Service  

                There are currently no active promotions other than the one's mentioned above. There is an option to give customers a discount of 5 percent but only if they sign up for a 2 year contract. This should only be offered to customers that are at risk of churning (churn_risk 1 or 2) or have very negative conversations.         
                """},
                {"role": "user", "content": f"{text}"}
            ]
        )

        output = {"recommendationText": completion.choices[0].message.content}

    if task == 'summarize':
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are helping a call center worker for a telco company called airwave. You will receive the user text content, which will include the call center worker and the customers spoken words converted to text. You are called at the end of a conversation to summarize the call. Make sure the summary is as short as possible but always includes all relevant parts of the conversation. You may receive information about the customer such as name and currently used products. The company has currently has 3 products: " + """
                AirSpeed Advanced
                High Speed Wireless Broadband

                Special Offer € 45 per month
                Cost is € 45 p/m for the first 3 months and € 60 thereafter
                12 Month contract
                Up to 70 Mbps / 7 Mbps

                FREE Fritzbox Router
                Installation fee of € 150
                Optional Home Phone Service

                and

                AirSpeed Plus
                High Speed Wireless Broadband

                Special Offer € 35 per month
                Cost is € 35 p/m for the first 3 months and € 50 thereafter
                12 Month contract
                Up to 50 Mbps / 5 Mbps

                FREE Fritzbox Router
                Installation fee of € 150
                Optional Home Phone Service

                and

                AirSpeed Home
                High Speed Wireless Broadband

                Special Offer € 25 per month
                Cost is € 25 p/m for the first 3 months and € 40 thereafter
                12 Month contract
                Up to 30 Mbps / 3 Mbps

                FREE Fritzbox Router
                Installation fee of €150
                Optional Home Phone Service

                Summarize the converastion short but include all important information.             
                """},
                {"role": "user", "content": f"{text}"}
            ]
        )

        output = {"recommendationText": completion.choices[0].message.content}

    if task == 'getCustomerInfo':
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": 'You are a helpful assistant for call center agents designed to analyze text from a conversation and figure out information about the customer who is calling outputing a JSON. Provide your answer in JSON structure like this {"name": "<The name of the customer>", "address": "<The street of the customer> <The house number of the customer as number, not string>", "dob": "<The date of birth of the customer as YYYY-MM-DD>". You will be given the entire conversation so far, do not worry if the information is not complete yet, just fill out whatever you can identify.'},
                {"role": "user", "content": f"{text}"}
            ]
        )

        info = json.loads(completion.choices[0].message.content)
        info_complete = False
        needed_informattion = ['name', 'address', 'dob']

        for info_elem in needed_informattion:
            if info_elem in info and info[info_elem] and info[info_elem] != "":
                info_complete = True
                if info_elem == 'dob' and not is_valid_date(info['dob']):
                    info_complete = False
            else:
                info_complete = False

        if info_complete:
            customer_info_from_db = retrieveCustomerInfo(
                info['name'], info['dob'], info['address'])
            if customer_info_from_db == False:
                output = {"recommendationText": completion.choices[0].message.content,
                          "foundCustomer": 0}
            else:
                output = {"recommendationText": completion.choices[0].message.content,
                          "foundCustomer": 1,
                          "customerInfo": customer_info_from_db}

        else:
            output = {
                "recommendationText": completion.choices[0].message.content}

    return output
