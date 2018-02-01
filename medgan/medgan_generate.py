from medgan import *
import os

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data = np.load(args.data_file)
    inputDim = data.shape[1]

    mg = Medgan(dataType=args.data_type,
                inputDim=inputDim,
                embeddingDim=args.embed_size,
                randomDim=args.noise_size,
                generatorDims=args.generator_size,
                discriminatorDims=args.discriminator_size,
                compressDims=args.compressor_size,
                decompressDims=args.decompressor_size,
                bnDecay=args.batchnorm_decay,
                l2scale=args.L2)

    # To generate synthetic data using a trained model:
    # Comment the train function above and un-comment generateData function below.
    # You must specify "--model_file" and "<out_file>" to generate synthetic data.
    mg.generateData(nSamples=1000,
                modelFile=args.model_file,
                batchSize=args.batch_size,
                outFile=args.out_file)
