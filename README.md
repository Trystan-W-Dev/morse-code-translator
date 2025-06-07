# morse-code-translator
Morse code translator with RNN, LSTM, and Transformer-based models

## Problem and Dataset Overview
For this project, I have attempted to recognize and translate Morse code audio files
into human readable text. This was an ideal problem to focus on because Morse is
a sequence of dots, dashes, and timing pauses and we were tasked with building a
Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) network,
and Transformer network to translate, generate, and predict a new sequence of
human readable words. The goal here for this sequence-to-sequence task was to
implement these different model architectures and compare the performance of
each.

The dataset used for this project consists of audio .wav files that make up the 100
most commonly used English words. The dataset was derived from the opensource
Morse Code Sound Collections library on Kaggle’s website. This dataset
was selected for this project due to containing short, predictable, and easy to
understand words that can be represented as sequences of characters along with its
relatively short size (100 words). It’s important to note that for real world natural
language processing tasks, a large and complex dataset size would be used instead
of a small dataset such as this one. This small dataset was chosen in order to be
within the scope of the project.

For preparing the data, each word was tokenized into individual characters so that
the models could learn the structure of each word carefully, and after tokenization
the models attempted to identify patters of characters and their associated
relationships. Mappings (i.e., character-to-index and index-to-character) were
implemented for converting individual characters to numbers and vice versa since
neural networks can’t read raw text. After mapping the dataset, padding was added
to small words for ensuring consistent input length. Truncation was also applied to
longer words for maintaining input length consistency. These techniques were used
on the dataset for ensuring uniform input sequence and model training efficiency.

## Summary of Models' Architecture
RNN
This model is implemented using Kera’s SimpleRNN layers which is just a basic
layer that processes an input sequence with one timestep at a time and is also used
for hidden RNN layers. There is an input layer that was used, and this layer
receives input (i.e., character sequences) and accepts shapes with timesteps and
features. Timesteps refers to the total number of characters in an input per word.
Features is the size of every character’s embedding.

For hidden layers, the total number used was 2. Each hidden layer contains 128
hidden units, and the parameter return sequences is set to True so that the layers
output a sequence instead of a single character prediction. For the dense layer, it
converts the RNN output to a series of probabilities over the output vocabulary by
using the SoftMax activation function. This function is applied at every timestep so
that it can predict the next character. For the lambda layer, it slices (or truncates)
the output so that it matches the sequence length requirement (i.e., 50 characters).
This model uses two hidden layers, and when built and compiled it uses a total of
55,845 parameters, which all of them can be used in model training.

LSTM
This model builds off the traditional RNN by using LSTM cells. The purpose of
this is to mitigate the known RNN vanishing gradient problem. For the input layer,
it receives character sequences similar as an RNN. For the hidden layers, the total
number used was 2 and they are bidirectional. Each hidden layer uses 128 LSTM
units. The bidirectional nature of these layers means that the units can read the
input sequence in a both forward and backward direction. This allows the LSTM
model to understand context better.

The hidden layers also use the parameter return sequences where its set to True so
that they can output a sequence instead of a single prediction, similarly like the
RNN. For the dense layer, it used a SoftMax activation function at each timestep of
the sequence. It is also wrapped in TimeDistributed so that it ensures all the
predictions are made per character. For the lambda layer, the output is truncated so
that it matches the sequence length requirement (i.e., 50 characters). Because this
model uses two hidden layers, when built and compiled it uses a total of 549,157
parameters, which all of them can be used in model training.

Transformer
This model uses self-attention mechanisms and does not rely on recurrence
compared to RNN and LSTM models. Because of this, transformers run in parallel
which makes them more effective at learning long sequences. For the input layer, it
receives character sequences the same way RNNs and LSTMs do. In the
Transformer encoder block (or hidden layer where computations and
transformations happen on data), it has a multi-head attention mechanism that
reads an input sequence in parallel heads. Each attention head learns about the
relationship between different characters. The attention output is added back into
the input and is normalized.

There is a feedforward network that is added here for processing the normalized
output. The feedforward network is made up of several dense layers and uses the
ReLU activation function. It also adds the regularization dropout technique. For the
lambda layer, it truncates the output to the first 50 timesteps like the RNN and
LSTM so that it matches the known sequence length requirement. There is a final
dense layer that uses the SoftMax activation function for predicting a probability
distribution over the character vocabulary at each given timestep.

## Implementation Notes and Training Observations
The RNN, LSTM, and Transformer models were implemented with TensorFlow
and Keras, and the model training included the following hyperparameter choices:
an epoch size of 20, a batch size of 32, the Adam optimizer with a default learning
rate (0.001), the sparse categorical cross entropy loss function, and accuracy
computed at each timestep for model training metrics. I found that setting the
epoch size to 20 gave the right balance that was needed for the models since any
larger epoch size of 20 introduced overfitting and any smaller numbers than 20
introduced underfitting. I noted that increasing or decreasing the epoch size of 20
by 2 (+2, -2) caused significant changes in the models training loss, validation loss,
training accuracy, and validation accuracy graph plots.

Using an epoch of 20 allowed the models to achieve convergence, but the first few
epoch iterations appeared to have made the loss drop drastically for the RNN and
LSTM models until the loss decrease became more stable as more epoch iterations
occurred. The Transformer model had a steady loss decrease from the first epoch to
the last. The RNN and LSTM accuracy plots also show that the train and validation
accuracy skyrocketed in the first few epoch iterations and then they leveled out
until the last epoch was reached. The Transformer accuracy plot showed gradual
increase in train and validation accuracy in the first few epochs but then
skyrocketed as the epoch iterations reached 50% completion.

The batch size of 32 was also a sweet spot for model training since it enabled the
RNN, LSTM, and Transformer models to generalize well on the test dataset. When
I tried increasing the batch size from 32 to 64, it decreased the model training time,
but it caused severe unstable weight updates which showed on the loss and
accuracy graphs. I also tried using a batch size of 16, and it showed more
fluctuations in the loss and accuracy graphs compared to using a batch size of 32.
The Adam optimizer was used because it is commonly used in training deep neural
networks and Transformer-based models. This optimizer adapts the learning rate
for each weight in the models, which enhances convergence.

The default learning rate was used with this optimizer which is 0.001. This
learning rate allows the models to learn slowly so that overshooting does not occur,
and it gives the models steady convergence. For the loss function, the sparse
categorical cross entropy function is used here since it is known for being memory
efficient when handling decent sized vocabularies. This is because it works directly
with integer indices, whereas the normal categorical cross entropy is used with
one-hot vectors and is often computationally expensive and slow. For metrics,
accuracy is used since it evaluates the model’s predictions at each epoch.

Because this project focused on translating Morse code audio files to human
readable text, in order for the RNN, LSTM, and Transformer models to be trained
on Morse, the audio files that were used for training had to be converted into inputs
that are required for model training. This part of the project is known as the feature
extraction pipeline. The Morse audio files are in .wav format, and they were
converted into a 1D time-domain signal. This was accomplished with the use of the
librosa audio processing library. The challenge here with loading multiple audio
files and converting them into 1D input vectors is every audio file has its own level
of noise and volume. For uniformity, a standard sample rate of 16,000 Hz was used
for resampling the .wav audio files.

This 16,000 Hz sample rate value was used because Morse signals can transmit at a
much lower frequency, so having a sampling rate value that is twice the frequency
(known as Nyquist Theorem in Radio Communications) used in Morse audio
transmissions provides sufficient audio coverage to capture the entirety of Morse
signals so that they can be completely reconstructed and that no audio features
would be missing. Various amplitudes in the .wav files were also normalized, or in
other words the amplitudes in the audio files were scaled to consistent volumes so
that it would enhance model training and prevent training instability. Next, for
feature extraction, the Mel-Frequency Cepstral Coefficients (MFCC) process is
used to convert the 1D time domain input signal into a 2D feature vector for
capturing spectral characteristics of the audio. This process also captures important
frequency features and can identify important information by using the Mel scale,
which compares perceived pitch with actual frequency.

## Quantitative and Qualitative Performance Analysis 
Quantitative
For quantitative analysis of this project, the metrics accuracy and loss were used to
show and compare model performances between training accuracy and loss versus
validation accuracy and loss. For the RNN model, the second epoch showed that
the training accuracy was 0.1259 and the training loss was 3.6074. The validation
accuracy was 0.9371 and the validation loss was 1.0600. The final epoch showed
that the training accuracy was 0.9346 and the training loss was 0.3319. The
validation accuracy was 0.9390 and the validation loss was 0.3128. The validation
values were slightly better overall, which means that the RNN model generalized
well to the test dataset.

For the LSTM model, the second epoch showed that the training accuracy was
0.4009 and the training loss was 3.0341. The validation accuracy was 0.9371 and
the validation loss was 0.5600. The final epoch showed that the training accuracy
was 0.9344 and the training loss was 0.2417. The validation accuracy was 0.9390
and the validation loss was 0.2340. Overall, the LSTM model validation values
were slightly better than the training values, which reflects that the model
generalized to the test dataset. Compared to the RNN model, the LSTM model has
the same validation accuracy values, but the LSTM validation loss is lower which
means it generalized better than the RNN.

For the Transformer model, the second epoch showed that the training accuracy
was 0.0023 and the training loss was 3.9161. The validation accuracy was 0.0057
and the validation loss 3.2218. The final epoch showed that the training accuracy
was 0.9313 and the training loss was 0.9900. The validation accuracy was 0.9371
and the validation loss was 0.9147. The Transformer model had significantly
higher training and validation losses compared to the RNN and LSTM models.
Because Transformers require large amounts of data for training, and my dataset
only included 100 common English words, I suspect this possibly caused the
Transformer model to underperform by not learning meaningful patterns in the
small dataset.

Qualitative
The qualitative analysis part of this project includes the predicted output
comparison between the RNN, LSTM, and Transformer models. After changing
hyperparameters several times to see how the models would react, the RNN and
LSTM models would sometimes give the right output (i.e., full words), sometimes
give mixed output (i.e., half completed words), and sometimes give the wrong
output, or no output at all. There was one instance where the Transformer model
gave the right output but at the expense of the RNN and LSTM models’ training
stability where they would not predict the right output. An example of what the
RNN model predicted is “OH”. An example of what the LSTM model predicted is
“OO”. With an epoch size of 20, a batch size of 32, and 2 hidden layers in each
model, the Transformer model did not give an output. However, when these
hyperparameter values are increased or sometimes decreased, the Transformer
model gives an output. When increasing the epoch size to 30 and decreasing the
batch size to 16, the RNN model gave the output “TOE”, the LSTM model gave
the output “SO” and the Transformer model gave the output “H”. When I increased
the epoch size to 40, the RNN model predicted “TO”, the LSTM model predicted
“TT”, and the Transformer model predicted “OO”.

## Key Takeaway and Challenge
One key takeaway from this project is the sequence models showed promising
signs of learning from the Morse audio file dataset. Given the models’ current state
and hyperparameter settings, they seldomly output coherent words with most of the
output resulting in incomplete words. But when number of epochs increases, and
batch size decreases, the models exhibit behavior where more complete words are
predicted. With more time and analysis of the current hyperparameters, it may be
possible to enable these models to increase the accuracy of their word predictions.

A key challenge here is finding the right balance of hyperparameter configurations
because you can change one hyperparameter thinking this will improve model
performance until you examine the model prediction output and the accuracy/loss
graphs and see that model training instability is introduced. Another challenge with
this project was dataset processing. Audio files have complex components to them
so trying to ingest audio files and converting them into input sequences that these
models can handle required tedious code implementation and changes. If not done
properly to where the models can extract meaningful patterns and features, it will
ruin the models’ training and performance.
