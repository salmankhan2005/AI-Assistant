import asyncio
import os
import subprocess
import webbrowser
from webbrowser import open as webopen
import keyboard  # type:ignore
import requests  # type:ignore
from AppOpener import close  # type: ignore
from AppOpener import open as appopen  # type: ignore
from AppOpener.features import AppNotFound  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from dotenv import dotenv_values  # type: ignore
from groq import Groq  # type: ignore
from pywhatkit import playonyt, search  # type: ignore
from rich import print

env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")


client = Groq(api_key=GroqAPIKey)

professional_responses: list = [
    "Your satisfaction is my top priority; feel free to reach out if there's anything else I can help you with.",
    "I'm at your service for any additional questions or support you may need don't hesitate to ask.",
]

messages = []

SystemChatBot = [
    {
        "role": "system",
        "content": f"Hello, I am {os.environ['Username']}, You're a contet writer. You have to write contet like letters, codes, applications, essays",
    }
]


def GoogleSearch(Topic):
    search(Topic)
    return True


def YoutubeSearch(query):
    Url4Search = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(Url4Search)
    return True


def Content(Topic: str):
    def OpenNotepad(File):
        default_text_editor = "notepad.exe"
        subprocess.Popen([default_text_editor, File])

    def ContetWriterAI(prompt):
        messages.append({"role": "user", "content": f"{prompt}"})
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None,
        )
        Answer = ""

        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        Answer = Answer.replace("</s>", "")

        messages.append({"role": "assistant", "content": Answer})
        return Answer

    Topic = Topic.replace("Content ", "")
    ContentByAI = ContetWriterAI(Topic)

    flname = rf"Data\{Topic.lower().replace(' ', '')}.txt"

    with open(flname, "w", encoding="utf-8") as file:
        file.write(ContentByAI)

    OpenNotepad(flname)
    return True


def PlayYoutube(query):
    playonyt(query)
    return True


def OpenApp(app, sess=requests.session()):
    try:
        appopen(app, match_closest=True, output=True, throw_error=True)
        return True
    except AppNotFound:

        def extract_links(html):
            if html is None:
                return []
            soup = BeautifulSoup(html, "html.parser")
            links = soup.find_all("a", {"target": "_self"})
            links = [link.get("href") for link in links]
            return links[0]

        def search_brave(query):
            url = f"https://search.brave.com/search?q={query}"
            head = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            }
            response = sess.get(url, headers=head)

            if response.status_code == 200:
                return response.text
            else:
                print("Failed to retrieve search results.")
            return None

        html = search_brave(app)

        if html:
            link = extract_links(html)
            if link:
                webopen(link)
                return True
            else:
                return False


def CloseApp(app):
    if "chrome" in app:
        pass
    else:
        try:
            close(app, match_closest=True, output=True, throw_error=True)
            return True
        except:  # noqa: E722
            return False


def System(command):
    def mute():
        keyboard.press_and_release("volume mute")

    def unmute():
        keyboard.press_and_release("volume mute")

    def volume_up():
        keyboard.press_and_release("volume up")

    def volume_down():
        keyboard.press_and_release("volume down")

    if command == "mute":
        mute()
    elif command == "unmute":
        unmute()
    elif command == "volume up":
        volume_up()
    elif command == "volume down":
        volume_down()
    return True


async def TranslateAndExecute(commands: list[str]):
    funcs = []

    for command in commands:
        fun = None  # Initialize fun before conditions

        if command.startswith("open "):
            if "open it" in command:
                pass
            elif "open file" == command:
                pass
            else:
                fun = asyncio.to_thread(OpenApp, command.removeprefix("open "))

        elif command.startswith("general "):
            pass  # Placeholder for general logic
        elif command.startswith("realtime "):
            pass  # Placeholder for realtime logic

        elif command.startswith("close "):
            fun = asyncio.to_thread(CloseApp, command.removeprefix("close "))

        elif command.startswith("play "):
            fun = asyncio.to_thread(PlayYoutube, command.removeprefix("play "))

        elif command.startswith("content "):
            fun = asyncio.to_thread(Content, command.removeprefix("content "))

        elif command.startswith("google search "):
            fun = asyncio.to_thread(
                GoogleSearch, command.removeprefix("google search ")
            )

        elif command.startswith("youtube search "):
            fun = asyncio.to_thread(
                YoutubeSearch, command.removeprefix("youtube search ")
            )

        elif command.startswith("system "):
            fun = asyncio.to_thread(System, command.removeprefix("system "))

        else:
            print(f"No Function Found for {command}")

        if fun:
            funcs.append(fun)

    # Gather all results asynchronously
    results = await asyncio.gather(*funcs)

    # Yield results as they come
    for result in results:
        if isinstance(result, str):
            yield result
        else:
            yield result


async def Automation(commands: list[str]):
    async for result in TranslateAndExecute(commands):
        print(result)
    return True


if __name__ == "__main__":
    asyncio.run(
        Automation(
            [
                "open facebook",
                "open instagram",
            ]
        )
    )
