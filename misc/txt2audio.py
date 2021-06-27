import argparse
from gtts import gTTS
import sys


def get_audio_from_txt(inputfile, outputfile, language='en'):
    print(f'reading from {inputfile}')
    with open(inputfile, 'r', encoding='utf-8') as file:
        file = file.read().replace('\n', " ")
        speaker = gTTS(text=str(file), lang=language, slow=False)
        speaker.save(outputfile)

def main():
    inputfile = ''
    outputfile = ''

    parser = argparse.ArgumentParser(description='takes .txt input and provides .mp3 output of file being read')
    parser.add_argument('i')
    parser.add_argument('o')

    args = parser.parse_args()
    inputfile = args.i
    outputfile = args.o

    print('Input file is "', inputfile)
    print('Output file is "', outputfile)

    if len(outputfile) < 1:
        outputfile = './audio.mp3'

    get_audio_from_txt(inputfile, outputfile, language='en')


if __name__ == "__main__":
   main()