from tkinter import Tk
from tkinter.filedialog import askopenfilename
import argparse
from gtts import gTTS
import sys

def get_audio_from_txt(input_file, output_file, language='en'):
    print(f'reading from {input_file}')
    with open(input_file, 'r', encoding='utf-8') as file:
        file = file.read().replace('\n', " ")
        speaker = gTTS(text=str(file), lang=language, slow=False)
        speaker.save(output_file)

def get_input_file():
    Tk().withdraw()
    filename = askopenfilename()
    return filename

def main():
    input_file = get_input_file()
    output_file = input('Please enter the name of the output file.\n'
                        'Do not add a file extension to it.\n')
    if not output_file.endswith('.mp3'): output_file += '.mp3'

    print(f'making {output_file}...')

    get_audio_from_txt(input_file, output_file, language='en')


if __name__ == '__main__':
    main()