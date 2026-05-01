### Preparations: Programming Project and Final Report
#### Instructions

Machine learning is a useful computational tool to deal with data problems that arise in our daily lives, industry, and scientific research. To demonstrate your understanding of the machine learning algorithms covered in this course, you are required to propose one programming task to use any machine learning algorithm to solve a real-world problem. You are highly recommended to propose a project that is closely related to your own research. In the following, three example projects are provided for your reference. You are also encouraged to use the materials on the Internet for reference, but you are required to write your own codes to implement your selected algorithm.
Project Examples

    Task 1: Bayesian learning for classifying Netnews text articles. Naive Bayes classifiers are among the most successful known algorithms for learning to classify text documents. We provide a dataset containing 20,000 newsgroup messages drawn from the 20 newsgroups. The dataset contains 1000 documents from each of the 20 newsgroups. Please download the zip file Download zip file for the dataDownload at this link. Then you are required to process the articles (such as using the bag-of-word model to represent the articles) and write a Naive Bayes' classifier to classify the news articles into 20 classes. You are required to perform the following subtasks.

        Perform class-vs-class classification, i.e., pairwise binary classification. Please compare the performance of the classifications for different pairs of classes, and analyze why some are good but some are not.
        Perform tri-class classification (i.e. three classes are involved). You can pick up at least five groups of tri-classes of data. Please compare the performance of the classifications for different groups of classes, and analyze why some are good but some are not.

    Task 2: Support vector machines for face recognition. Face recognition is a learning problem that has recently received a lot of attention. One standard approach involves reducing the dimensionality of the problem using PCA and then selecting the nearest class (eigenfaces). Support Vector Machines (SVM) are becoming very popular in the machine learning community as a technique for tackling high-dimensional problems. Can SVMs outperform standard face recognition algorithms? Please implement the SVM algorithm by yourself. Download the experimental dataset Download experimental dataset if this interests you.
    Task 3: Neural network learning to recognize faces. A neural network learning algorithm called Back-propagation is among the most effective approaches to machine learning when the data includes complex sensory input such as images. You can download the data set Download data set with over 600 face imagesDownload here. The data set contains face images from 20 different persons with different emotions and poses. Write your codes to implement the neural network, by which you are expected to classify the emotions. That is, you are expected to classify the images into four classes: neutral, happy, sad, angry. Then compare the results between the following two cases: (1) include the face images with sunglasses in training and (2) not include face images with sunglasses in training, and analyze your results.

#### Deliverables and Submissions

##### Project proposal (30 points, due in Week 4)

The project proposal is where you declare your plan to complete this assignment. For your selected project, the proposal should not exceed 1 page and cover the following:

    Problem statement - identifying the problem and motivation to solve it
    Solution - solution description, what area of machine learning and what machine learning algorithm the solution will utilize, why this can't be done with traditional methods, and input/output expectation
    Data - what data you have found related to this problem, if any, or your approach to acquiring data
    Timeline - The expected tasks to complete and weekly progress expectation

Please refer to the rubric for grading on the submission page of this task. 

##### Progress update (10 points, due in Week 6)

The progress report should update the instructors and the class on your progress and update the project timeline and expectations accordingly. The progress report is a presentation where students cover:

    Data - you should have finalized the dataset and completed any necessary data cleaning
    Feature learning - you should have tested out a few feature extraction methods
    Model - you should have a preliminary model with some of the features in the data
    Timeline - you should have an understanding of how well your time estimates have been and update the timeline accordingly
    Project expectation - you should update the project expectation based on the progress you've made so far
    Constraints - you should mention any constraints that have risen and any decisions you've made to mitigate them including any ethical implications of your application

Your progress report should be submitted to the submission page of this task.

##### Final deliverables (60 points, due in Week 8)

The final report should cover the following:

    Problem Statement - what the problem is from the customer's perspective
    Solution - what your tool/product/model does to solve the customer's problem
    Demo - a demo of your product working; this can be a recorded video or a live demo
    Assumptions, Constraints & Implications - what assumptions you're making, any constraints that you're dependent on, and what implications your solution results in
    How your solution was built - what data you used, feature learning you applied, and model you trained
    Summary - Summarize the problem, solution, and issues

Besides the final report, you are also required to submit a code package of your implementations of your selected algorithm and how to use it to manipuate your selected dataset. Implementations in C, C++, Java, Python, and Matlab will be accepted. If you would like to use another programming language, please first check with the instructor or the TA. Points will be taken off for failure to comply with this requirement.

The project should be submitted via Blackboard. Submit a ZIPPED directory called project.zip (no other forms of compression accepted, contact the instructor if you do not know how to produce .zip files). The directory should contain source codes and your report. Including binaries that work on windows (for Java and C++) is optional. The submission should also contain a file called readme.txt, which should specify precisely:

    Name and CSM ID of the student.
    What programming language is being used.
    How the code is structured.
    How to run the code, including very specific compilation instructions, if compilation is needed. Instructions such as "compile using g++" are NOT considered specific.

Insufficient or unclear instructions will be penalized by up to 20 points.

Please refer to the rubric for grading the submission page of this task.

| Module | Task | Points |
|---|---|---|
| Module 1 | Start thinking about and researching what programming project you want to complete in this course | — |
| Module 2 | Continue to think about and research your programming project of choice | — |
| Module 3 | Narrow down your choices for your programming project of choice | — |
| Module 4 | Propose your programming project for the final report | 30 |
| Module 5 | Begin working on the final report | — |
| Module 6 | Provide an update on your progress towards the final report | 10 |
| Module 7 | Work on the final report | — |
| Module 8 | Submit your final report | 60 |
| **Total** | | **100** |