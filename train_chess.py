import gradio as gr
import fastbook
from fastbook import *

chess_pieces = ['pawn', 'bishop', 'knight', 'rook', 'queen', 'king']
chess_colors = ['white', 'black']

path = Path('chess_pieces')

# If the path for the chess pieces doesn't exist, we don't have data. So create the directories
# and download all the images needed from Duck Duck Go
if not path.exists():
    print('making chess piece directory...')
    path.mkdir()
    for color in chess_colors:
        for piece in chess_pieces:
            print(f'making {color} {piece} directory')
            dest = (path/f'{color}_{piece}')
            dest.mkdir(exist_ok=True)
            results = search_images_ddg(f'photo of a single {color} {piece} chess piece', 15)
            download_images(dest, urls=results)

fns = get_image_files(path)
print(fns)

failed = verify_images(fns)
print(failed)

# If some of the images were malformed or otherwise broken, get rid of them
failed.map(Path.unlink)

# Create the template needed to split up the data into chunks. We will then use a dataloader to feed
# these blocks into the trainer
chess_pieces_data = DataBlock(
    # Define the input and output types
    blocks=(ImageBlock, CategoryBlock),

    # Define the function needed to get the data
    get_items=get_image_files,

    # Split the data into training and validation sets
    splitter=RandomSplitter(valid_pct=0.2, seed=42),

    # Define the function needed to label the data
    get_y=parent_label,

    # Transform the data into standardized sizes
    item_tfms=Resize(128))

# Create our DataLoaders class so we can add some DataLoader objects to it
# These will handle the data augmentation and loading into RAM
# Uses the defaults of the past iteration of the DataLoaders object
dls = chess_pieces_data.dataloaders(path)

# Define a new type of DataLoaders
chess_pieces_data = chess_pieces_data.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

# Now that we've defined it again, we reload it
dls = chess_pieces_data.dataloaders(path)

# Now we pass in the DataLoaders into the trainer and fine tune our model
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(8)

# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix()
#
# interp.plot_top_losses(5, nrows=1)

# Finally, export our model so we can use it as a "function" for applications later
learn.export()