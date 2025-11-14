"""
Qwen Text Embedding Examples
Comprehensive examples for all major use cases

SETUP:
1. Install dependencies: uv pip install dashscope python-dotenv numpy scikit-learn
2. Create .env file with: DASHSCOPE_API_KEY=your_key_here
3. Run: python examples.py
"""

import os
from http import HTTPStatus
import numpy as np
from dotenv import load_dotenv

# Note: Import dashscope only when API key is configured
# Uncomment below when ready to use:
# import dashscope


class QwenEmbeddingExamples:
    """Collection of Qwen embedding usage examples"""

    def __init__(self):
        """Initialize and load API key from environment"""
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not found in environment.\n"
                "Please create a .env file with: DASHSCOPE_API_KEY=your_key_here"
            )

        # Uncomment when dashscope is imported
        # dashscope.api_key = api_key

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def example_1_semantic_search(self):
        """Example 1: Semantic Search"""
        print("=" * 60)
        print("Example 1: Semantic Search")
        print("=" * 60)

        query = "什么是人工智能？"
        documents = [
            "人工智能是计算机科学的一个分支",
            "机器学习是实现人工智能的重要方法",
            "今天天气很好"
        ]

        # NOTE: Uncomment below when dashscope is imported
        """
        # Get query embedding
        query_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=query,
            dimension=1024
        )

        # Get document embeddings
        doc_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=documents,
            dimension=1024
        )

        # Calculate similarities
        query_emb = query_resp.output['embeddings'][0]['embedding']
        for i, doc_emb in enumerate(doc_resp.output['embeddings']):
            similarity = self.cosine_similarity(query_emb, doc_emb['embedding'])
            print(f"[{similarity:.4f}] {documents[i]}")
        """

        print(f"Query: {query}")
        print("Documents:", documents)
        print("\nNote: Uncomment API call code to run actual search")

    def example_2_recommendation(self):
        """Example 2: Recommendation System"""
        print("\n" + "=" * 60)
        print("Example 2: Recommendation System")
        print("=" * 60)

        user_history = ["科幻电影", "动作片"]
        items = ["科幻大片", "浪漫爱情", "悬疑推理"]

        # NOTE: Uncomment below when dashscope is imported
        """
        # Build user profile
        history_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=user_history,
            dimension=1024
        )
        history_embs = [emb['embedding'] for emb in history_resp.output['embeddings']]
        user_profile = np.mean(history_embs, axis=0)

        # Get item embeddings
        items_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=items,
            dimension=1024
        )

        # Calculate recommendations
        recommendations = []
        for i, item_emb in enumerate(items_resp.output['embeddings']):
            score = self.cosine_similarity(user_profile, item_emb['embedding'])
            recommendations.append((items[i], score))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        print("Recommendations:")
        for item, score in recommendations:
            print(f"  [{score:.4f}] {item}")
        """

        print(f"User history: {user_history}")
        print(f"Items: {items}")
        print("\nNote: Uncomment API call code to run actual recommendations")

    def example_3_classification(self):
        """Example 3: Zero-Shot Classification"""
        print("\n" + "=" * 60)
        print("Example 3: Zero-Shot Classification")
        print("=" * 60)

        text = "这件衣服质量很好"
        labels = ["数码产品", "服装配饰", "食品饮料"]

        # NOTE: Uncomment below when dashscope is imported
        """
        # Get embeddings
        all_texts = [text] + labels
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=all_texts,
            dimension=1024
        )

        text_emb = resp.output['embeddings'][0]['embedding']
        best_label = None
        best_score = -1

        for i, label in enumerate(labels):
            label_emb = resp.output['embeddings'][i + 1]['embedding']
            score = self.cosine_similarity(text_emb, label_emb)
            if score > best_score:
                best_score = score
                best_label = label

        print(f"Text: {text}")
        print(f"Classification: {best_label} (confidence: {best_score:.4f})")
        """

        print(f"Text: {text}")
        print(f"Labels: {labels}")
        print("\nNote: Uncomment API call code to run actual classification")

    def example_4_clustering(self):
        """Example 4: Text Clustering"""
        print("\n" + "=" * 60)
        print("Example 4: Text Clustering (requires scikit-learn)")
        print("=" * 60)

        texts = [
            "人工智能发展",
            "机器学习研究",
            "NBA总决赛",
            "世界杯足球"
        ]

        # NOTE: Uncomment below when dashscope and sklearn are imported
        """
        from sklearn.cluster import KMeans

        # Get embeddings
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=texts,
            dimension=1024
        )
        embeddings = np.array([emb['embedding'] for emb in resp.output['embeddings']])

        # Cluster
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters = {0: [], 1: []}
        for text, label in zip(texts, labels):
            clusters[label].append(text)

        print("Clusters:")
        for cluster_id, cluster_texts in clusters.items():
            print(f"  Cluster {cluster_id}: {cluster_texts}")
        """

        print(f"Texts: {texts}")
        print("\nNote: Uncomment API call code to run actual clustering")

    def example_5_anomaly_detection(self):
        """Example 5: Anomaly Detection"""
        print("\n" + "=" * 60)
        print("Example 5: Anomaly Detection")
        print("=" * 60)

        normal_samples = ["会议成功", "项目顺利", "功能完成"]
        test_samples = ["工作进展好", "asdfghjkl", "垃圾内容"]

        # NOTE: Uncomment below when dashscope is imported
        """
        # Get normal sample embeddings
        normal_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=normal_samples,
            dimension=1024
        )
        normal_embs = [emb['embedding'] for emb in normal_resp.output['embeddings']]
        normal_center = np.mean(normal_embs, axis=0)

        # Test samples
        test_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=test_samples,
            dimension=1024
        )

        threshold = 0.6
        print("Anomaly detection results:")
        for i, test_emb in enumerate(test_resp.output['embeddings']):
            similarity = self.cosine_similarity(test_emb['embedding'], normal_center)
            is_anomaly = similarity < threshold
            status = "ANOMALY" if is_anomaly else "NORMAL"
            print(f"  [{status}] {test_samples[i]} (similarity: {similarity:.4f})")
        """

        print(f"Normal samples: {normal_samples}")
        print(f"Test samples: {test_samples}")
        print("\nNote: Uncomment API call code to run actual detection")

    def example_6_advanced_features(self):
        """Example 6: Advanced Features (text_type, hybrid search)"""
        print("\n" + "=" * 60)
        print("Example 6: Advanced Features")
        print("=" * 60)

        # NOTE: Uncomment below when dashscope is imported
        """
        # 1. Using text_type
        query_emb = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input="深度学习",
            text_type="query",  # Optimized for queries
            dimension=1024
        )

        doc_emb = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input="深度学习是机器学习的子领域",
            text_type="document",  # Optimized for documents
            dimension=1024
        )

        # 2. Using hybrid search
        hybrid_resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input="Python机器学习",
            output_type="dense&sparse",  # Both dense and sparse vectors
            dimension=1024
        )

        dense_vec = hybrid_resp.output['embeddings'][0]['embedding']
        sparse_vec = hybrid_resp.output['embeddings'][0]['sparse_embedding']

        print("Dense vector dimensions:", len(dense_vec))
        print("Sparse vector non-zero elements:", len(sparse_vec['indices']))
        """

        print("Advanced features demonstrated:")
        print("  1. text_type parameter (query vs document)")
        print("  2. Hybrid search (dense + sparse vectors)")
        print("\nNote: Uncomment API call code to run actual examples")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Qwen Text Embedding Examples")
    print("=" * 60)
    print("\nSetup Instructions:")
    print("1. Install: uv pip install dashscope python-dotenv numpy scikit-learn")
    print("2. Create .env file with: DASHSCOPE_API_KEY=your_key_here")
    print("3. Uncomment API call code in each example function")
    print("4. Uncomment 'import dashscope' at the top of this file")
    print("\n" + "=" * 60)

    try:
        examples = QwenEmbeddingExamples()

        # Run all examples
        examples.example_1_semantic_search()
        examples.example_2_recommendation()
        examples.example_3_classification()
        examples.example_4_clustering()
        examples.example_5_anomaly_detection()
        examples.example_6_advanced_features()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease set up your API key first:")
        print("1. Create a .env file in this directory")
        print("2. Add: DASHSCOPE_API_KEY=your_actual_api_key")


if __name__ == "__main__":
    main()
