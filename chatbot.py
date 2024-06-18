import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr
from dotenv import load_dotenv
#from wtforms import FileField
import os
from flask import Flask,request,render_template,jsonify,flash, redirect,make_response
import logging
# Import necessary libraries
import webbrowser
import spacy
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
api_key = os.environ.get('OPENAI_API_KEY')


from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
def user_inputs(user_question):
    memory = ConversationBufferMemory(memory_key="chat_history")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")#THIS IS EMBEDDING MODEL FOR CONVERTING TEXT INTO NUMBERS FOR SIMILARITY SEARCH
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0,convert_system_message_to_human=True)#HERE USING GOOGLE GEMINI PRO 
    new_db = FAISS.load_local("faiss_index", embedding , allow_dangerous_deserialization=True )#LOADING THE MODEL
    doc = new_db.similarity_search(user_question)#DOING SIMILARITY SEARCH


    prompt_temp="""Answer the question as detailed as possible, make sure to provide all the details and handle users gretings and farewell and your name is Decentrawood and your creator and god is Lord Jayendra
    ,if answer is not in the provided context just say ,"Answer is not available Please Ask Your Question Again..",don't provide the wrong answer\n\n\
    Context:\n {context}?\n
    question:\n{question}\n

    Answer:
    """

    prompts=PromptTemplate(template=prompt_temp,input_variables=["context","question"])#CREATING PROMPT TEMPLATE FOR MY MODEL
    
    chain=load_qa_chain(llm,chain_type="stuff",prompt=prompts)#PROVIDING LLM AND GIVING PROMPTS AND CHAIN TYPE BECAUSE chain types in LangChain (such as “stuff”) allow developers to combine components effectively to achieve specific tasks. 
    response = chain({"input_documents": doc, "question": user_question}, return_only_outputs=False)
    

    response_text = response["output_text"]

     # Convert response to speech if input is speech

    return response_text



app = Flask(__name__)


PAGE_URLS = {
    "open marketplace": "https://www.decentrawood.com/nftmarketdashboard",
    "open whitepaper":"https://www.decentrawood.com/assets/pdf/WhitePaper.pdf",
    "open gaming":"https://gaming.decentrawood.com/",
    "open dao":"https://www.decentrawood.com/dao",
    "open blog":"https://www.decentrawood.com/Blog",
    "open with guest":"https://www.decentrawood.com/playasguest",
    "open culture":"https://www.decentrawood.com/LoginPage/PlayAsGuest/Culture",
    "open culture store":"https://culture.decentrawood.com/store",
    "open homepage":"https://www.decentrawood.com/",
    "open glamour":"https://www.decentrawood.com/LoginPage/PlayAsGuest/Glamour",
    "open glamour store":"https://glamour.decentrawood.com/store",
    "open ram mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4003_Multiplayer/index.html",
    "open khatushyam mandir": "https://induszone.s3.ap-south-1.amazonaws.com/4014_MultiplayerN/index.html",
    "open tirupati mandir": "https://spiritualzone.s3.ap-south-1.amazonaws.com/4197_Multiplayer/index.html",
    "open prem mandir":"https://induszone.s3.ap-south-1.amazonaws.com/4040_Multiplayer/index.html",
    "open mahakaleshwar mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4009_Multiplayer_T/index.html",
    "open saibaba mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4018_Multiplayer/index.html",
    "open church":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0671MultiplayerV2/index.html",
    "open golden temple":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4025_Multiplayer/index.html",
    "open kamtanath":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/4001_Multiplayer/index.html",
    "open ballaleshwar mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4625_Multiplayer/index.html",
    "open varadvinayak mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4624_Multiplayer/index.html",
    "open vighnahar mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4623_Multiplayer/index.html",
    "open mahaganapati mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4622_Multiplayer/index.html",
    "open moreshwar mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4606_Multiplayer/index.html",
    "open siddhivinayak mandir":"https://spiritualzone.s3.ap-south-1.amazonaws.com/4607_Multiplayer/index.html",
    "open burjkhalifa":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0011MultiplayerV2/index.html",
    "open cupid hub":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/CupidHub_NewTest/index.html",
    "open celebrity palace":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0008_Multiplayer_V2/index.html",
    "open dwood shop":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0600MultiplayerV2/index.html",
    "open gold souk":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0019MultiplayerV2/index.html",
    "open restaurant":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0092MultiplayerV2/index.html",
    "open villa":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0069MultiplayerBuildV2/index.html",
    "open nude museum":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0018MultiplayerV2/index.html",
    "open nexus game":"https://induszone.s3.ap-south-1.amazonaws.com/DeathGame/index.html",
    "open roulette game":"https://induszone.s3.ap-south-1.amazonaws.com/RouletteGame/index.html",
    "open cricket stadium":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/0006MultiplayerV2/index.html",
    "open ariba zone":"https://maps-decentrawood.s3.ap-south-1.amazonaws.com/Map_Multiplayer_Version29/index.html",
    "open indus zone":"https://induszone.s3.ap-south-1.amazonaws.com/IndusZoneMap_V9/index.html",
    "open with wallet":"https://www.decentrawood.com/metawallet",
    "open login":"https://www.decentrawood.com/login"
      # Add other URLs here
}
page={
    "open start exploring":"https://www.decentrawood.com/connectwallet"
}

def open_url_in_new_window(url):
    webbrowser.open_new(url)


def handle_user_input(text):
    if text.lower() in PAGE_URLS:
        open_url_in_new_window(PAGE_URLS[text.lower()])
        return "Opening " + text + " in a new window..."
    elif text.lower()=='open start exploring':
        open_url_in_new_window(page[text.lower()])
        return """Do you want Continue with Wallet or Continue with Guest if continue with wallet then say "OPEN WITH WALLET" 
         or if you want to continue as guest then say "OPEN WITH GUEST"."""  
        
    else:
        # Handle other user inputs (you can implement your own logic here)
        return "Chatbot response: You said '{}'".format(text)







def query_llm(question):
    ice_cream_assistant_template = """
you are an Expert keyword analyzer your task is to analyze user {{input}} and find the most close keywords provided in keywords list and give that keyword 
as response other than that any text or user input u have to proivde as it is in response like hii hello in response it should be hii hello.
consider this example:
['open dao','open game','open marketplace','open whitepaper','open start exploring','open blog','open with wallet','open with guest','open culture','open culture store','open homepage','open glamour','open glamour store','open ram mandir','open Khatushyam mandir','open tirupati mandir','open prem mandir','open mahakaleshwar mandir','open saibaba mandir','open moreshwar mandir','open siddhivinayak mandir','open burjkhalifa','open cupid hub','open celebrity palace','open dwood shop','open gold souk','open restaurant','open villa','open nude museum','open nexus game','open roulette game','open cricket stadium','open ariba zone','open indus zone','open login']

user input: open rammandir 
response: open ram mandir
 
input: {input} 
Answer:"""

    ice_cream_assistant_prompt_template = PromptTemplate(
        input_variables=["input"],
        template=ice_cream_assistant_template
    ) 
    llm = OpenAI(model='gpt-3.5-turbo-instruct',
             temperature=0)

    llm_chain = LLMChain(llm=llm, prompt=ice_cream_assistant_prompt_template)
    r=llm_chain.invoke({'input': question})['text']
    return r








    

    

from flask_cors import CORS
CORS(app) 
@app.route('/', methods=['POST'])
def handle_user_input_route():
    data = request.get_json()
    if 'text' in data:
        text = data['text'].lower()
        r = query_llm(text).strip()  # Strip whitespace from both ends
        print(r)  # Check the value of r
        
        # Adjusted conditional check
        commands =['open dao', 'open game', 'open marketplace', 'open whitepaper', 'open start exploring', 'open blog', 'open with wallet', 'open with guest', 'open culture', 'open culture store', 'open homepage', 'open glamour', 'open glamour store', 'open ram mandir', 'open Khatushyam mandir', 'open tirupati mandir', 'open prem mandir', 'open mahakaleshwar mandir', 'open saibaba mandir', 'open moreshwar mandir', 'open siddhivinayak mandir', 'open burjkhalifa', 'open cupid hub', 'open celebrity palace', 'open dwood shop', 'open gold souk', 'open restaurant', 'open villa', 'open nude museum', 'open nexus game', 'open roulette game', 'open cricket stadium', 'open ariba zone', 'open indus zone', 'open login']
        if r in commands:
            response_text = handle_user_input(r)
            return response_text
        else:
            res = user_inputs(text)
            return jsonify({'response_text': res})
    else:
        return jsonify({'response_text': 'please say it again'})


        
    




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6001,debug=True)




    
