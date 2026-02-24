import asyncio
from youtube_converter import YoutubeConverter
from youtube_chatbot import run_youtube_chatbot
from youtube_summarizer import youtube_summarizer_tunnel
from prompts import prompt_templates



#Example on how to run youtube summarizer tunnel 
#user_video = input("Please insert a url: ")
    #asyncio.run(youtube_summarizer_tunnel(url=user_video,prompt_templates=prompt_templates))
    
if __name__ == "__main__":
    user_video = input("Please insert a url: ")
    print("Please wait a few moments for completion")
    #if tunnel == True, generate html, pptx, and chatbot = True
    
    if asyncio.run(youtube_summarizer_tunnel(url=user_video,prompt_templates=prompt_templates)):
        user_input = input("What would you like to do next?: Convert to Slides, View HTML, or Chat with the content? (Type 'slides', 'html', or 'chat')").lower()
        youtube_converter = YoutubeConverter()  
        if user_input == "slides":
            print("Generating slides...")
            # Code to convert summary to slides
            youtube_converter.create_pptx_from_data()
        elif user_input == "html":
            print("Generating HTML...")
            # Code to generate HTML
            youtube_converter.create_html_from_data()
        elif user_input == "chat":
            print("Launching chatbot...")
            # Code to launch chatbot interface
            run_youtube_chatbot()
            
            

