# Vocab repetition app

This is an AI-powered app for flashcard vocab repetition in optimal intervals for language learners.

It's already usable, but mind that it is still a work in progress and will come with a more user-friendly interface in the future, a broader README file with more in-depth explanation of the model.

## Spaced repetition

Spaced repetition is a method of learning where newly introduced and more difficult flashcards are shown more frequently, while older and less difficult flashcards are shown less frequently. It has been proven to increase the rate of learning compared to an approach of evenly-spaced repetitions.

Usually spaced repetition algorithms estimate the likelihood that the user will remember the answer, and when that likelihood is below a given threshold - usually 90% - the item is repeated. Wrong answers will cause the next interval to be shorter, while correct answers will lengthen it.

Popular applications for flashcard spaced repetition learning include Memrise, Anki or SuperMemo (a pioneer in the field).

My approach is to use recursive neural networks for calculating recall likelihood, and therefore optimal intervals, based on previous answers to the item. The app shows flashcards in both ways, as active and passive memory are different. 


## Usage

To run the app you simply need to type in the following command

```bash
py -3.12 main.py course_name
```

Where course_name is the name of the .xlsx file with flashcards in the /wordlists folder.

The app will then present you with flashcards - click enter when you think you know the answer (meaning of the word in the second language), then the answer will be shown - if you guessed correctly, press 3, otherwise press 2.

The /wordlists folder contains a few of my own lists which you can use to see how to create your own wordlist.

## Calculating optimal intervals

When you revise words, there are records stored in the /records folder, which, for every pair of words and their order, contains the information about the time of repetition (in UNIX time) and whether you got it right (1) or wrong (0).

The network model used is LSTM with three nodes at the input and one at the output - first two nodes correspond to whether you got it right or wrong, the third node is log of the time interval since last repetition, and the output node is the likelihood that you get it right the next time.

A separate model is trained for each course.