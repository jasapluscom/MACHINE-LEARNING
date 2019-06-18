#!/usr/bin/env python3
'''
assistant.py
sample application
for python training at www.jasaplus.com
'''
from selenium import webdriver
import time, os, inspect, sys, re
import speech_recognition as sr
from gtts import gTTS
import aiml
from bs4 import BeautifulSoup


#start vars
BRAIN_FILE = "assistant.brain"
COMMAND_FILE = "command.dump"
#eof vars

#start class
class Assistant():
    def speech(self, data):
        try:
            tts = gTTS(text=data, lang='en', slow=True).save("tmp.mp3")
            cmd = "mpg123 tmp.mp3"
            pipe = os.popen(cmd)
        except:
            raise

    def setopt(self):
        opt = webdriver.ChromeOptions()
        opt.add_argument('--alsa-input-device=plughw:1,0')
        opt.add_argument('--mute-audio')
        opt.add_experimental_option("prefs", { \
        "profile.default_content_setting_values.media_stream_mic": 1,
        "profile.default_content_setting_values.notifications": 1
        })
        return opt

    def prepare_kernel(self, FILE, aiml_path):
        k = aiml.Kernel()
        try:
            if os.path.exists(FILE):
                k.loadBrain(FILE)
            else:
                k.bootstrap(learnFiles=aiml_path)
                k.saveBrain(FILE)
        except:
            raise
        return k

    def prepare_selenium_browser(self):
        browser = None
        try:
            opti = self.setopt()
            browser = webdriver.Chrome(chrome_options = opti)
            browser.get("http://localhost/webspeech/")
            browser.set_window_position(800, 370)
            browser.set_window_size(520, 350)
        except:
            raise

        return browser

    def log(self, data):
        try:
            with open('output.txt', 'w') as file:
                file.write(data)
        except:
            raise

    def main_listen_process(self, browser, k, cmd):
        try:
            count = 0
            data_prev = ""
            self.speech("Ok sir I am ready")
            while True:
                data = browser.find_element_by_id("data").get_attribute("value");
                if data != "None":
                    data = data.strip()
                    if (data != data_prev) and (len(data) > 0):
                        data_prev = data
                        print(data)
                        response = cmd.respond(data)
                        if (response.find("ok sir") == -1) and (response.find("The time") == -1):
                            response = k.respond(data)
                            print(response)
                            self.speech(response)
                        count = 0
                time.sleep(1)
                count+=1
                if (count > 15):
                    startme = browser.find_element_by_id("startme")
                    if startme != None:
                        startme.click()
        except:
            raise
#eof class

if __name__ == "__main__":
    assistant = Assistant()
    browser = assistant.prepare_selenium_browser()
    kernel = assistant.prepare_kernel(BRAIN_FILE, "aiml/*.aiml")
    command = assistant.prepare_kernel(COMMAND_FILE, "aiml_commands/commands.aiml")
    assistant.main_listen_process(browser, kernel, command)
