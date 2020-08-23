import argparse
import model


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # input argument
  parser.add_argument(
      "--batch_size",
      help = "The number of images to use per training step.",
      type = int,
      default = 32
  )
  parser.add_argument(
      "--train_data_path",
      help = "Path of the datasets used for training.",
      type = str,
      default = None
  )
  parser.add_argument(
      "--val_data_path",
      help = "Path of the datasets used for validation.",
      type = str,
      default = None
  )
  parser.add_argument(
      "--output_dir",
      help = "Path to save the model.",
      type = str,
      default = None
  )
  parser.add_argument(
      "--image_shape",
      help = "The image size (height and width and num_channels) used for training.",
      type = tuple,
      default = (224, 224, 3)
  )
  parser.add_argument(
      "--num_epochs",
      help = "The number of steps that the training job will run.",
      type = int,
      default = 5
  )
  args = parser.parse_args()
  params = args.__dict__

  model.train_and_evaluate(params)
