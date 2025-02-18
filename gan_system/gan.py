import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class GANSystem:
    def __init__(self, latent_dim: int = 100):
        """
        Initialize the GAN system.
        :param latent_dim: Dimension of the latent space for the generator.
        """
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan_model = self.build_gan_model()

    def build_generator(self) -> Sequential:
        """
        Build the generator model.
        """
        model = Sequential(name="Generator")
        model.add(Dense(128, input_dim=self.latent_dim, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(768, activation='tanh'))  # Output similar to BERT embeddings size
        logger.debug("Generator built.")
        return model

    def build_discriminator(self) -> Sequential:
        """
        Build the discriminator model.
        """
        model = Sequential(name="Discriminator")
        model.add(Dense(512, input_dim=768, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug("Discriminator built and compiled.")
        return model

    def build_gan_model(self) -> Sequential:
        """
        Combine generator and discriminator into a GAN model.
        """
        self.discriminator.trainable = False
        model = Sequential(name="GAN")
        model.add(self.generator)
        model.add(self.discriminator)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        logger.debug("GAN model built and compiled.")
        return model

    def train(self, real_data: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """
        Train the GAN model.
        :param real_data: Real data for training.
        :param epochs: Number of training epochs.
        :param batch_size: Size of each training batch.
        """
        if len(real_data) == 0:
            logger.warning("No valid data for GAN training. Skipping...")
            return

        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise)
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            if real_data.shape[0] != batch_size:
                real_data = np.tile(real_data, (batch_size // real_data.shape[0] + 1, 1))[:batch_size]

            d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_data, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan_model.train_on_batch(noise, valid_labels)
            logger.info(f'GAN Training Epoch {epoch + 1}/{epochs}: d_loss={d_loss}, g_loss={g_loss}')

    def generate_creative_output(self) -> np.ndarray:
        """
        Generate creative output using the trained generator.
        """
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        creative_embedding = self.generator.predict(noise)
        logger.info("Creative output generated.")
        return creative_embedding
