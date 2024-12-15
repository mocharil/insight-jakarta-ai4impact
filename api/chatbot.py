from utils.knowledge import ChatSystem

# Initialize chat system
chat_system = ChatSystem()

# Initialize chat
chat_system.initialize_chat()


import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os, time, re
from urllib.parse import quote
TOKEN = '7349422721:AAGukNj65e583aJBHnrpWy2-ri_4nJ0gymc'

def sendMessage(pesan, chat_id, reply_id='', TOKEN=TOKEN):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&parse_mode=MarkdownV2&text={pesan}&reply_to_message_id={reply_id}'
    response = requests.get(url)
    sending_message = response.json()
    return sending_message

def getUpdates(TOKEN, offset):
    url = f'https://api.telegram.org/bot{TOKEN}/getUpdates'
    response = requests.get(f"{url}?offset={offset}")
    return response.json()

replied = []

while True:
    all_message = getUpdates(TOKEN, 0)
    for result in all_message.get('result',[]):
        offset = result['update_id'] + 1
        message = result['message']
        start_time = time.time()

        date = datetime.fromtimestamp(message['date'])
        reply_id = message['message_id'] 
        chat_id = message['from']['id']

        range_time = (datetime.now()-date).total_seconds()/60
        if range_time > 10:
            continue
        if reply_id in replied:
            continue


        print('chat id:', chat_id)
        try:
            username = message['from']['username']
        except:
            username = chat_id
        print('username:', username)

        text = message.get('text', message.get('caption', ''))
        print('text:', text)
        print('reply id:', reply_id) 
        print('date:', date)       


        if text == '/start':   
            is_start_command = True
            bot_response = f"""Halo! 👋
    Saya *JakSee*, asisten virtual Insight Jakarta. Tanyakan apa saja tentang Jakarta, dan saya siap membantu Anda!
    Apa yang ingin Anda ketahui?"""
            
        else:
            bot_response = chat_system.ask(text)['answer']  # Will use knowledge-base
            bot_response = re.sub(r'(\s)\*(\s)',r"\1", bot_response, flags = re.I|re.S)
            bot_response = re.sub(r'\*{2,}',"*", bot_response)

                
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        doc = {
            "user_id": str(chat_id),
            "username": username,
            "message_id": str(reply_id),
            "message_text": text,
            "bot_response": bot_response,
            "response_time_ms": response_time_ms,
            "timestamp": (datetime.now() + timedelta(hours = 7)).isoformat(),
            "answer": bot_response
        }

        markdown_v2_escapes = ['_', '[', ']', '~', '`', '>', '#', '+',
                               '|', '{', '}', '!', '.', '-', '(', ')']
        escaped_response = re.sub(f"({'|'.join(re.escape(c) for c in markdown_v2_escapes)})", r'\\\1', bot_response)
        response_encoded = quote(escaped_response)   
        sendMessage(response_encoded, chat_id, reply_id, TOKEN=TOKEN)
        print('-----------------------\n')
        replied.append(reply_id)
            
        time.sleep(1)
    time.sleep(3)