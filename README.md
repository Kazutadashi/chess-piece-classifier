# Chess Piece Classifier
This program was created to classify images of singular chess pieces based off images of that chess piece. For example, if you take a picture of a Pawn, and feed it into the model, you will get the probability the model thinks its a Pawn, along with its guess as to what the piece actually is. This model was created using the fastai library, and fine tuned on resnet18
### How to use
Visit the live interface over at ![Huggingface Spaces](https://huggingface.co/spaces/Kazutadashi/chess-piece-classifier) and upload an image of a chess piece. After submitting you will be presented with your results. 

# About This Project
This project is a modification of Chapter 2 from the ![fastai Course](https://course.fast.ai/Lessons/lesson2.html). It was initially designed as a bear, black bear, or teddy bear classifier, but things were modified to classify chess pieces instead. In doing this, I learned quite a bit and I would like to save these lessons learned here in this repo for future reference.

### The Beginning
Initially, I only downloaded 100 images from DuckDuckGo for chess different type of chess piece, that is, 100 images of black pawns, black knights, white rooks, etc. From the course I learned that it is often a good idea just to try and train the model without any data cleaning, because your results may already be quite powerful. Unfortunately in my case, this was not true:

```
epoch  train_loss  valid_loss  error_rate  time
0      3.761093     2.230717   0.677885    00:28

epoch  train_loss  valid_loss  error_rate  time
0      2.796414    2.080197    0.644231    00:39
1      2.579284    1.874652    0.567308    00:39
2      2.280658    1.874342    0.567308    00:39
3      2.041709    1.833892    0.557692    00:39
4      1.840056    1.789111    0.548077    00:41
5      1.695149    1.781452    0.562500    00:42
6      1.559488    1.778758    0.562500    00:43
7      1.448577    1.774907    0.552885    00:44
```
As we can see, I was getting an error rate of over 55%! Not very helpful. I then drilled down and looked to see what exactly might be going wrong here using a confusion matrix:
<p align="center">
  <img src="https://github.com/Kazutadashi/chess-piece-classifier/assets/40162378/341fad27-c6f7-4ff6-a740-f375b691dedd" alt="Confusion Matrix"/>
</p>

There seemed to be mistakes all around unfortunately. However looking at some of the actual predictions did reveal some potential issues:
<p align="center">
  <img src="https://github.com/Kazutadashi/chess-piece-classifier/assets/40162378/68f4ff6c-7597-4117-953a-250405501f6f" alt="Predictions"/>
</p>

The data was quite different than what I was picturing, and the bad images, especially those that had multiple pieces in one picture, were severely damaging the results. This was definitely a time where some cleaning was in order. I went in and manually moved all the images I felt would train the wrong concepts. Things like multiple pieces in one image, very esoteric specialty pieces, and some random stuff that almost wasn't even related to chess, were all moved into a new folder called "bad_images":

![image](https://github.com/Kazutadashi/chess-piece-classifier/assets/40162378/832f91c7-d06b-4d51-80dd-b65f7e9638b4)

After everything was cleaned up, I did a final once-over and removed any duplicates or ones I had missed giving us these final amounts:
![image](https://github.com/Kazutadashi/chess-piece-classifier/assets/40162378/6d60b859-3c1d-4893-ae84-6dc293a8fcc9)

So how much of a difference did this actually make to the results? Quite a bit actually! I retrained the model and gave it another go and got these results:
```

epoch  train_loss  valid_loss  error_rate  time
0      3.790416    2.288522    0.713376    00:22
epoch  train_loss  valid_loss  error_rate  time
0      2.552994    1.784612    0.598726    00:28
1      2.206371    1.328237    0.426752    00:28
2      1.868055    1.071951    0.324841    00:28
3      1.565837    0.903450    0.292994    00:28
4      1.337779    0.829002    0.248408    00:29
5      1.164122    0.792078    0.235669    00:29
6      1.044300    0.793201    0.216561    00:29
7      0.957329    0.788493    0.222930    00:29

```

Let's take a look at the confusion matrix once again:
<p align="center">
  <img src="https://github.com/Kazutadashi/chess-piece-classifier/assets/40162378/19e95745-8768-4ae7-a3d4-2ef496fc29b7", alt="Final Confusion Matrix"/>
</p>
The white queen and black bishop are definitely struggling still, but overall we have much better performance.
## Interesting Findings
One thing I noticed when testing this out in production with random images from the internet and from my own sets, was the surprising errors to the color of the piece. I had initially thought that the color would be almost always correct, but the various shapes could be causing issues. I suspect that this is due to the random resized cropping that was being done during data augmentation, as some images have completely white backgrounds, even though the piece is black. 

### Biased Data
Another interesting note was the problem of biased data. A lot of chess sets have brown pieces for the black pieces, instead of just pure black. However there was very little training data to reflect this. Anytime I uploaded an image like this, the model struggled heavily. I suppose this is due to the idea of "out of domain data" that was mentioned in this chapter of the book, but it was interesting to see how profound of an effect this had. 

## Final Notes
I believe I could get better results with different types of data augmentation, and with the inclusion of more varied data, however I've decided to put this project to rest for now and continue on with the course. Maybe one day after I've learned more I can come back to this and try and go for even better accuracy. We shall see!
