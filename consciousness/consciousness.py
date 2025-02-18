import time
import networkx as nx
from sklearn.cluster import KMeans
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, GlobalAveragePooling1D, Dense
from collections import defaultdict
import logging
from gan_system.gan import GANSystem
from dqn_system.dqn import DQNSystem
from latent_reasoning.latent_module import LatentReasoningModule
from memory.memory import Memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class ConsciousnessSystem:
    def __init__(self):
        """
        Initialize the consciousness system.
        """
        self.sensorial_memory = Memory("Sensorial")
        self.operational_memory = Memory("Operational")
        self.transitory_memory = Memory("Transitory")
        self.long_term_memory: dict = defaultdict(list)
        self.priority_memory = Memory("Dynamic Priorities")
        self.conclusion_buffer = Memory("Conclusion Buffer")
        self.episodic_memory = Memory("Episodic Memory")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')

        self.conclusion_model = self.build_conclusion_model()
        self.emotion_model = self.build_emotion_model()

        self.gan_system = GANSystem(latent_dim=100)
        self.dqn_system = DQNSystem(state_size=768, action_list=["explore", "focus_safety", "neutral"])

        self.dependency_graph = nx.DiGraph()
        self.latent_reasoning_module = LatentReasoningModule(hidden_dim=768, iterations=3)

        self.sandbox_update_interval = 5  # Seconds between sandbox updates

        logger.info("Consciousness system initialized.")

    def build_conclusion_model(self) -> Model:
        """
        Build the conclusion model.
        """
        input_layer = Input(shape=(None, 768), name="ConclusionInput")
        x = MultiHeadAttention(num_heads=4, key_dim=64, name="Conclusion_MultiHeadAttention")(input_layer, input_layer)
        x = GlobalAveragePooling1D(name="Conclusion_Pooling")(x)
        x = Dense(64, activation='relu', name="Conclusion_Dense")(x)
        output = Dense(1, activation='sigmoid', name="Conclusion_Output")(x)
        model = Model(inputs=input_layer, outputs=output, name="ConclusionModel")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug("Conclusion model built and compiled.")
        return model

    def build_emotion_model(self) -> Model:
        """
        Build the emotion model.
        """
        input_layer = Input(shape=(None, 768), name="EmotionInput")
        x = MultiHeadAttention(num_heads=4, key_dim=64, name="Emotion_MultiHeadAttention")(input_layer, input_layer)
        x = GlobalAveragePooling1D(name="Emotion_Pooling")(x)
        x = Dense(64, activation='relu', name="Emotion_Dense")(x)
        output = Dense(5, activation='softmax', name="Emotion_Output")(x)
        model = Model(inputs=input_layer, outputs=output, name="EmotionModel")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.debug("Emotion model built and compiled.")
        return model

    def preprocess_data(self, raw_data: str) -> np.ndarray:
        """
        Preprocess raw text data into embeddings.
        :param raw_data: Raw input text.
        """
        inputs = self.tokenizer(raw_data, return_tensors='tf', truncation=True, padding=True)
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state.numpy()
        logger.debug("Data preprocessed into embeddings.")
        return embeddings

    def cluster_data(self, data: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """
        Cluster data using K-Means.
        :param data: Input data for clustering.
        :param n_clusters: Number of clusters.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)
        logger.debug("Data clustered using K-Means.")
        return labels

    def generate_conclusion(self, context: str) -> str:
        """
        Generate a conclusion based on the input context.
        :param context: Input context.
        """
        embeddings = self.preprocess_data(context)
        refined_embeddings = self.latent_reasoning_module.refine(embeddings)
        prediction = self.conclusion_model.predict(refined_embeddings)
        conclusion = "Positive" if prediction[0][0] > 0.5 else "Negative"
        self.conclusion_buffer.store(conclusion)
        logger.info(f"Conclusion generated: {conclusion} for context: {context}")
        return conclusion

    def simulate_emotions(self, context: str) -> str:
        """
        Simulate emotions based on the input context.
        :param context: Input context.
        """
        emotions = ["Happiness", "Sadness", "Anger", "Surprise", "Neutral"]
        embeddings = self.preprocess_data(context)
        refined_embeddings = self.latent_reasoning_module.refine(embeddings)
        prediction = self.emotion_model.predict(refined_embeddings)[0]
        dominant_emotion = emotions[np.argmax(prediction)]
        logger.info(f"Emotion simulated: {dominant_emotion} for context: {context}")
        return dominant_emotion

    def update_priority_memory(self) -> None:
        """
        Update dynamic priority memory based on conclusions.
        """
        conclusions = self.conclusion_buffer.retrieve()
        for conclusion in conclusions:
            if conclusion == "Positive":
                self.priority_memory.store("Increase exploration")
            elif conclusion == "Negative":
                self.priority_memory.store("Focus on safety")
        self.conclusion_buffer.clear()
        logger.debug("Dynamic priorities updated.")

    def consolidate_long_term_memory(self, data: Any, category: str) -> None:
        """
        Consolidate data into long-term memory.
        :param data: Data to store.
        :param category: Category for storage.
        """
        self.long_term_memory[category].append(data)
        logger.info(f"Data consolidated into long-term memory under '{category}': {data}")

    def store_episodic_event(self, event_description: str) -> None:
        """
        Store an episodic event in memory.
        :param event_description: Description of the event.
        """
        timestamp = time.time()
        event = (timestamp, event_description)
        self.episodic_memory.store(event)
        logger.info(f"Episodic event stored: {event}")

    def manage_dependencies(self, task_id: str, dependencies: List[str]) -> None:
        """
        Manage task dependencies.
        :param task_id: ID of the task.
        :param dependencies: List of dependent tasks.
        """
        self.dependency_graph.add_node(task_id)
        for dep in dependencies:
            self.dependency_graph.add_edge(dep, task_id)
        logger.info(f"Task '{task_id}' set with dependencies: {dependencies}")

    def resolve_dependencies(self, task_id: str) -> bool:
        """
        Resolve task dependencies.
        :param task_id: ID of the task.
        """
        predecessors = list(self.dependency_graph.predecessors(task_id))
        if not predecessors:
            return True
        resolved = all(
            any(str(dep) in str(event)
                for events in self.long_term_memory.values()
                for event in events)
            for dep in predecessors
        )
        return resolved

    def get_ready_tasks(self) -> List[str]:
        """
        Get tasks ready for execution.
        """
        ready_tasks = [node for node in self.dependency_graph.nodes if self.resolve_dependencies(node)]
        logger.info(f"Ready tasks: {ready_tasks}")
        return ready_tasks

    def state_to_vector(self, state_str: str) -> np.ndarray:
        """
        Convert a state string into a vector representation.
        :param state_str: State description.
        """
        embeddings = self.preprocess_data(state_str)
        vector = np.mean(embeddings, axis=1)
        return vector

    def update_sandbox_memory(self) -> None:
        """
        Update the sandbox memory periodically.
        """
        logger.info("Updating sandbox memory...")
        for i in range(3):
            time.sleep(self.sandbox_update_interval)
            logger.info(f"Sandbox update: Iteration {i+1} completed.")
        logger.info("Sandbox memory update completed.")

    def produce_output(self) -> str:
        """
        Produce output based on consolidated long-term memory.
        """
        aggregated = " | ".join(f"{key}: {values}" for key, values in self.long_term_memory.items())
        output = f"Output generated using consolidated memory:\n{aggregated}"
        logger.info("Output produced from sandbox.")
        return output

    def simulate_consciousness(self, raw_inputs: List[str]) -> None:
        """
        Simulate the consciousness system.
        :param raw_inputs: List of raw input data.
        """
        logger.info("=== Start Consciousness Simulation ===")

        # Phase 1: Collect sensory data
        for input_data in raw_inputs:
            try:
                embeddings = self.preprocess_data(input_data)
                self.sensorial_memory.store(embeddings)
                self.store_episodic_event(f"Sensory input received: {input_data}")
            except Exception as e:
                logger.error(f"Error preprocessing '{input_data}': {e}")

        # Phase 2: Transfer data to operational memory
        operational_data = self.sensorial_memory.retrieve()
        self.operational_memory.store(operational_data)

        # Phase 3: Generate conclusions and emotions
        for input_data in raw_inputs:
            conclusion = self.generate_conclusion(input_data)
            emotion = self.simulate_emotions(input_data)
            self.transitory_memory.store((conclusion, emotion))
            self.store_episodic_event(f"Conclusion for '{input_data}': {conclusion}, emotion: {emotion}")

        # Phase 4: Update dynamic priorities
        self.update_priority_memory()
        priorities = self.priority_memory.retrieve()
        logger.info(f"Dynamic priorities updated: {priorities}")

        # Phase 5: Consolidate data into long-term memory
        for idx, input_data in enumerate(raw_inputs):
            category = f"Cluster_{idx % 3}"
            self.consolidate_long_term_memory(input_data, category)

        # Phase 6: Integrate creativity via GAN
        if raw_inputs:
            try:
                embeddings_list = [self.preprocess_data(data) for data in raw_inputs]
                all_embeddings = np.vstack(embeddings_list)
                real_data = np.mean(all_embeddings, axis=1)

                if len(real_data) == 0:
                    logger.warning("No valid data for GAN training. Skipping...")
                else:
                    if real_data.shape[0] < 32:
                        real_data = np.tile(real_data, (32 // real_data.shape[0] + 1, 1))[:32]
                    self.gan_system.train(real_data, epochs=10, batch_size=32)
                    creative_output = self.gan_system.generate_creative_output()
                    logger.info(f"Creative output generated: {creative_output}")
            except Exception as e:
                logger.error(f"Error during GAN processing: {e}")

        # Phase 7: Dynamic feedback via DQN
        state_str = "current_context"
        state_vec = self.state_to_vector(state_str)
        action = self.dqn_system.choose_action(state_vec)
        reward = 1 if action == "explore" else -1
        next_state_str = "updated_context"
        next_state_vec = self.state_to_vector(next_state_str)
        done = False
        self.dqn_system.train(state_vec, action, reward, next_state_vec, done)

        # Phase 8: Manage task dependencies
        task_id = "Task_1"
        dependencies = ["Task_0"]
        self.manage_dependencies(task_id, dependencies)
        if self.resolve_dependencies(task_id):
            logger.info(f"Task '{task_id}' has no pending dependencies. Proceeding...")
        else:
            logger.info(f"Task '{task_id}' waiting for dependency resolution.")

        logger.info("=== End Consciousness Simulation ===")
        self.update_sandbox_memory()
