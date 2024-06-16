### A Priori Algorithm 연관규칙분석

- 경영학에서 **Market Basket Analysis 장바구니 분석**으로 알려진 방법론으로, 소비자들의 구매 이력 데이터를 토대로 “X 아이템을 구매하는 고객들은 Y 아이템 역시 구매할 가능성이 높다”는 **규칙**을 도출해내는 알고리즘으로, **Contents-based recommendation 컨텐츠 기반 추천**의 기본이 되는 방법론이다.
- 규칙의 효용성은 Support(지지도), Confidence(신뢰도), Lift(향상도) 지표를 반영하여 평가한다. 기존에는 가능한 모든 경우의 수를 탐색하여 효용성이 높은 규칙을 찾아내는 방식을 이용하였다. 하지만, 해당 방식은 아이템 수가 증가할수록 계산에 소요되는 시간이 기하급수적으로 증가하게 된다.  
- A Priori Algorithm은 규칙을 만들때, **Frequent item sets 빈발 집합**만을 고려하여 연관 규칙을 생성하는 알고리즘이다. 간단히 설명하면, 아이템 집합 A의 support인, P(A)가 0.1보다 작다면, 아이템 A를 포함한 집합들의 support는 P(A)를 넘지 못한다. 왜냐하면, A가 단독으로 등장할 확률인 P(A)는 A와 B가 동시에 나타날 확률인 P(A, B)보다는 크거나 같을 것이기 때문이다. 따라서, P(A)가 효용성이 있다고 생각하는 기준을 넘지 못할 경우, 아이템 A를 포함한 집합은 계산에서 제외할 수 있다.
![image](https://github.com/All4Nothing/recommendation-system/assets/81239098/bad76387-19ad-424f-8ee4-ea045251fce4)

***Reference***  
*[연관규칙분석(A Priori Algorithm)](https://ratsgo.github.io/machine%20learning/2017/04/08/apriori/)*

### Collaborative Filtering 협업 필터링

Collaborative Filtering은 사용자들의 과거 행동 이력을 분석하여 유사한 취향을 가진 다른 사용자들의 취향을 예측하는 방법론

- User-based Collaborative Filtering
    - 사용자 간의 유사성을 기반으로 추천
    - 예를 들어, 사용자 A와 B가 비슷한 아이템을 좋게 평가했을 경우, A가 좋아할 아이템을 B가 좋아한 아이템과 유사한 아이템을 추천할 수 있음
- Item-based Collaborative Filtering
    - 아이템 간의 유사성을 기반으로 추천
    - 사용자가 이전에 평가한 아이템과 유사한 다른 아이템을 추천
- 유사도 측정에는 Cosine Similarity(코사인 유사도), Pearson Correlation Coefficient(피어슨 상관계수), Euclidean Distance(유클리드 거리) 등이 사용됨

다음과 같은 과정으로 아이템을 추천한다.
1. User-Item Matrix 생성
2. User 간의 유사도 계산
3. 아이템 예상 평점 추론
    1. 사용자가 보지 않은 영화를 본 사용자들의 평점을 추출
    2. 사용자와의 유사도 계산
    3. 각 사용자 별 ‘유사도 X 평점’ 계산
      $$sim(u,u') \times R_{u'i}$$
    4. 유사도가 반영된 가중치 평점을 합산
      $$\sum_{u'}sim(u,u') \times R_{u'i}$$
    5. 가중치를 나누어 평균 평점을 계산하여, 사용자가 보지 않은 영화의 평점을 추론
      $$\^R_{ui} = \frac{\sum_{u'}sim(u,u') \times R_{u'i}}{\sum_{u'}|sim(u,u')|}$$

***Reference***  
*[Collaborative Filtering with Spark](https://www.slideshare.net/MrChrisJohnson/collaborative-filtering-with-spark)*  
*[Collaborative Filtering](https://medium.com/@toprak.mhmt/collaborative-filtering-3ceb89080ade)*  

### FP-Growth(Frequent Patterns Growth)  
- Apriori 알고리즘을 개선한 알고리즘으로, 계산 속도 및 BigData Scale 병렬 처리에 효과적인 알고리즘
- Apriori 알고리즘에서 candidate를 만들고, support를 구하기 위해 DB를 스캔하는 횟수는 최대 가장 긴 transaction의 아이템 수까지 일어날 수 있다. FP-Growth에서는 DB를 스캔하는 횟수는 단 2번이며, candidate를 만들지 않는다. 대신 Tree와 Node Link라는 특별한 자료구조를 사용한다.

***Reference***  
*[FP-Growth 알고리즘 개념 정리](http://blog.naver.com/PostView.nhn?blogId=sindong14&logNo=220661064114&parentCategoryNo=&categoryNo=48&viewDate=&isShowPopularPosts=true&from=search)*

### Matrix Factorization

- Recommedation problem을 matrix의 비어있는 부분의 값을 예측하는 문제로 바꿔서 생각하면 다음과 같은 식으로 문제를 나타낼 수 있다.
    
    $$min_{R^{hat}}||R^{hat}-R||^2_F$$
    
    - R은 비어있는 곳이 없는 원래의 데이터, R^hat은 비어있는 곳을 복구한 데이터*
    
- 두 matrix의 RMSE를 계산하는 것이 이 문제의 objective funtion이 되며, matrix를 완성시키는 문제라는 의미에서 Matrix Completion이라고 부른다.
- Matrix Completion 문제를 풀기 위한 가장 우수한 성능을 보이는 것으로 알려진 방법이 Matrix Factorization이다.

***Reference***

*[Machine Learning 스터디 (17) Recommendation System (Matrix Completion)](http://sanghyukchun.github.io/73/)*  

### Item2Vec

**Paper : Item2Vec: Neural Item Embedding for Collaborative Filtering**  https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf

- Word2Vec을 이용해 아이템을 vector화 시킨 item-based CF
- Single item recommendations이 user-to-item recommendations보다 좋은 성능을 보임
- User-item CF와 다르게, 아이템 간의 관계를 느슨하게 정의한게, 관계를 직접 학습하도록 최적화하는 것보다 성능이 좋았음
- Item2Vec : apply SGNS to item-based CF

***Reference***

*[Skip-Gram with Negative Sampling, SGNS](https://wikidocs.net/69141)*

*[우선 Word2Vec기술에 대하여 알아보고 추천 시스템에서의 item2vec에 대하여 알아보자](https://brunch.co.kr/@goodvc78/16)*

### Deep Learning Approach

**Paper : Deep Neural Networks for YouTube Recommendations** 

https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf

Youtube 비디오 추천을 위한 Deep Neural Network architecutre

시스템 구조는 크게 candidate generation과 ranking 2가지로 나뉜다.

1. Candidate generation : 추천할 후보 비디오를 몇 백개 내외로 뽑아냄
- recommendation problem을 extreme multiclass classification으로 재정의
- 학습 과정에서 explict 정보(ex. ’좋아요’ 정보)는 사용하지 않고, implict 정보(비디오를 끝까지 시청했는지)만을 사용
- 트레이닝 데이터 특성상 ML을 단순하게 적용하면 오래된 아이템들이 더 추천을 많이 받게 된다. 이를 해결하기 위해 Example Age(비디오의 나이)를 input으로 넣어준다.
1. Ranking : 그 중에서 순위를 매겨 추천
- Deep Neural Network를 이용해 비디오와 사용자의 관계를 계산

use the implict feedback of watches

***Reference***

*[Machine Learning 스터디 (17) Recommendation System (Matrix Completion)](http://keunwoochoi.blogspot.com/2016/09/deep-neural-networks-for-youtube.html)*

### Wide & Deep Model

**Paper : Wide & Deep Learning for Recommender Systems  https://arxiv.org/abs/1606.07792**

- Recommender system에서는 Memorization과 Generalization 모두 중요하다.
- Wide Model : Memorization된 값을 통해서 추천 (ex. 아아를 검색한 사용자가 쿠키를 시킨 기록이 많았다면, 아아를 검색한 사용자에게 쿠키를 추천하게 됨) → 기존에 기억된 결과로만 추천
- Deep Model : 아이템을 일반화 시켜서 추천 (ex. 아아를 일반화(커피)하여, 라떼를 추천) → 지나치게 일반화(under filtering)되는 문제가 발생할 수 있음(ex. 아아를 검색한 사람에게 따뜻한 라떼를 추천)
- 두 모델을 결합한 Wide & Deep Model 제안
- Google Play store에서 높은 효율 개선을 보임

***Reference***

*[Wide and deep network 모델 활용하기](https://bcho.tistory.com/1187)*

### Factorization Machine(FM)

Matrix factorization(MF)은 다차원의 matrix 데이터를 낮은 차원의 matrix들로 분해하여 데이터의 숨겨진 패턴을 발견하고 예측하는 데 사용된다. 주로 collaborative filtering, 이미지 처리, 자연어 처리 등에서 사용될 수 있다.

Factorization Machine은 Big Sparse Matrix에서 특히 유리하며, 다양한 유형의 데이터에서 변수 간 상호 작용을 모델링 하는데 중점을 둔다.

Recommender system에서 MF는 사용자-아이템 데이터만 가지고, latent feature를 학습하는 알고리즘이라, side-features를 사용할 수 없다. 예를 들어, 영화 추천 시스템에서 영화의 장르, 영화 감독과 같은 정보를 이용할 수 없다. 하지만 FM에서는 이용할 수 있다.

MF는 사용자의 개입이 필요한 협업 필터링 접근 방식으로, “Cold-start Problem”이라 불리는 문제에서는 작동하지 않는다. 예를 들어, 새로운 영화가 나왔을 때, 영화를 본 사용자가 아무도 없다면, MF는 추천 시스템으로서 작동할 수 없다. 하지만, 영화의 대한 다른 정보(장르, 감독, 배우 등)과 같은 정보가 있어 FM은 kick-start할 수 있다.

***Reference***

*[Difference Between Factorization Machines and Matrix Factorization ?](https://stats.stackexchange.com/questions/108901/difference-between-factorization-machines-and-matrix-factorization)*
