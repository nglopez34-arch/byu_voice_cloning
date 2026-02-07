Welcome to the byu speech cloning TTS respository! You can use this code to clone any voice found on speeches.byu.edu.
This is done by fine-tuning a pre-existing TTS model on a specific voice.
You can either choose your own voice that you would like to clone, or use a pre-existing fine-tuned model found in the models folder.
These instructions will walk you through the former option. If you would like to do the latter, do step 1 and then skip to step number 7.
These instructions were written assuming that you are working out of a terminal rather than an IDE.
It will be easiest to follow along if you use a terminal, and simply run the commands that I tell you to.

0. Before starting:
    To get an idea of what kind of audio this code can produce, listen to LISTEN_TO_ME.wav

1. Setup:
    I built this repository on a computer that uses Linux (Ubuntu 24.04.3 LTS) and has an nvidia GPU.
    If your setup is different, good luck. I'm sure you can do it but you may have to do some things differently.
    You're going to run this code in a virtual environment using an older version of python (3.11). Here's how to set it up:
        a. Check to see if you have python 3.11:
            python3.11 --version
        b. If you don't, install like this or in a different way:
            sudo apt update
            sudo apt install python3.11 python3.11-venv
        c. Navigate to the project directory, and then create and enter the virtual environment like this:
            python3.11 -m venv venv
            source venv/bin/activate
    This project requires specific versions of packages to be installed. To go about this the easy way, just run:
        pip install -r requirements.txt
    If all of this worked, you should now be ready to run the code.

2. Data acquisition
    Now it's time to decide who's voice you're going to clone. Go to https://speeches.byu.edu/speakers/ and
    pick somebody. If you want the audio to have decent quality, you're going to want to pick someone with several speeches.
    The goal is to have about 2 hours of audio for training data. You can probably get away with less.
    The less you use, the less the model will sound like the person being cloned.
    Edit fetch_audio.py by pasting the url for the author that you wish to clone right after the imports. That's all you have to change:
        nano fetch_audio.py
    This should be the only file that you have to edit.
    Run the program:
        python fetch_audio.py
    You should see all of the audio clips show up in a new directory named "your_author_here"_speeches
    Because some of the speakers voices change over the decades as they age, I recommend deleting the speeches where they are either very young or very old.
    You want the voice to sound about the same in all of the speeches.

3. Trim audio clips
    Now here's the tedious part. You're likely going to have to trim a bunch of the intros off of the beginning of the audio clips.
    You don't want to train the model on any other voices than the intended one. This will confuse it and lead to poor quality results.
    For most but not all of the speeches the person who is conducting the meeting reads an introduction at the beginning.
    You can do this using any audio editor, but I like audacity:
        sudo apt update
        sudo apt install audacity

4. Speech transcription
    Now we need to transcribe all of the speeches so that we have .txt files to use during training.
    For this, we will use a model named Whisper. But whisper performs poorly with long audio clips.
    This means that we have to break down all of the speeches into 5-20 second sections.
    This script will do this, using multiple CPU cores in parallel:
        python segment_audio.py
    Now that the clips have been broken down into approximately sentence sizes, we can transcribe (this uses gpu):
        python transcribe.py

5. Dataset preparation.
    Now comes the final step in creating a dataset for training and evaluation:
        python prepare_dataset.py

6. Fine-tuning existing model
    I suggest reading through this whole section before you spin up your GPU fans. After you do, this is how you start training:
        python finetune_xtts.py
    Training proceeds in discrete steps. At each step, the model processes a small batch of audio–text pairs, predicts the corresponding speech representation, compares that prediction to the real audio, and computes a loss value that measures how wrong the prediction was.
    The optimizer then slightly adjusts the model’s parameters to reduce that error. Every once in a while, the trainer saves a checkpoint of the model and generates test audio using predefined sentences.
    These test sentences are logged to TensorBoard under the Audio tab, where a slider allows you to listen to how the model sounds at different training steps.
    You will want to monitor TensorBoard while training. This is how you access it if you are cloning Elder Holland:
        tensorboard --logdir ~/PycharmProjects/tts_model/training_output_jeffrey_r_holland
    Then you will click on this link that pops up, it will take you to the tensorboard:
        http://localhost:6006/
    To monitor training, you should run TensorBoard while the script is running and regularly listen to the generated audio samples. Early checkpoints usually sound generic or unstable; later ones should gradually resemble the target speaker more closely.
    Training should be stopped when the voice sounds most natural and accurate, even if the numerical loss continues to decrease. The “best” model is not always the final checkpoint.
    Instead, you select the checkpoint whose generated audio sounds best to your ears and use that checkpoint for inference. In practice, listening to checkpoint audio is the primary way to judge training quality and detect overfitting.
    If this sounds like too much to ask, don't worry. You can just run it until the loss starts to go up, and then stop the script. When you generate audio in the next script, it will automatically pick the model it thinks is best.


7. Generate audio
    Execute this script to generate audio. The audio will both be played in real time, and saved to a folder for your enjoyment.
    If you do not like the way that the audio file sounds, you may execute the same script with the same text again and get a different audio clip.
    This is because there is randomness inserted at various points within the model.
        python generate_audio.py


