from transcripy import Transcriber

def main():
    # create transcriber with base model
    transcriber = Transcriber(model_size="base")
    
    # example: transcribe audio file
    # text = transcriber.transcribe("example.mp3")
    # print(text)
    
    # example: transcribe video file
    # text = transcriber.transcribe("example.mp4")
    # print(text)
    
    # example: save to file
    # transcriber.transcribe("example.mp3", output_file="output.txt")
    
    print("transcripy example - uncomment lines to use")

if __name__ == "__main__":
    main()

