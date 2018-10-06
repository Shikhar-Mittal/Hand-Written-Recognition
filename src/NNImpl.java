import java.util.*;

/**
 * The main class that handles the entire network Has multiple attributes each
 * with its own use
 */

public class NNImpl {
	private ArrayList<Node> inputNodes; // list of the output layer nodes.
	private ArrayList<Node> hiddenNodes; // list of the hidden layer nodes
	private ArrayList<Node> outputNodes; // list of the output layer nodes

	private ArrayList<Instance> trainingSet; // the training set

	private double learningRate; // variable to store the learning rate
	private int maxEpoch; // variable to store the maximum number of epochs
	private Random random; // random number generator to shuffle the training
							// set

	/**
	 * This constructor creates the nodes necessary for the neural network Also
	 * connects the nodes of different layers After calling the constructor the
	 * last node of both inputNodes and hiddenNodes will be bias nodes.
	 */

	NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random,
			Double[][] hiddenWeights, Double[][] outputWeights) {
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;
		this.random = random;

		// input layer nodes
		inputNodes = new ArrayList<>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();
		for (int i = 0; i < inputNodeCount; i++) {
			Node node = new Node(0);
			inputNodes.add(node);
		}

		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);

		// hidden layer nodes
		hiddenNodes = new ArrayList<>();
		for (int i = 0; i < hiddenNodeCount; i++) {
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for (int j = 0; j < inputNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);

		// Output node layer
		outputNodes = new ArrayList<>();
		for (int i = 0; i < outputNodeCount; i++) {
			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for (int j = 0; j < hiddenNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}
			outputNodes.add(node);
		}
	}

	/**
	 * Get the prediction from the neural network for a single instance Return
	 * the idx with highest output values. For example if the outputs of the
	 * outputNodes are [0.1, 0.5, 0.2], it should return 1. The parameter is a
	 * single instance
	 */

	public int predict(Instance instance) {
		ArrayList<Double> listDbl = new ArrayList<Double>();
		listDbl = instance.attributes;

		ArrayList<Integer> listInt = new ArrayList<Integer>();
		listInt = instance.classValues;

		forwardPass(listDbl);
		double min = Double.MIN_VALUE;
		int count = 0;
		for (int i = 0; i < listInt.size(); i++) 
		{
			if (outputNodes.get(i).getOutput() > min) 
			{
				min = outputNodes.get(i).getOutput();
				count = i;
			}
		}
		return count;
	}

	/**
	 * Train the neural networks with the given parameters
	 * <p>
	 * The parameters are stored as attributes of this class
	 */

	public void train() {
		int epochCount = 0;

		while (epochCount < this.maxEpoch)
		{
			Collections.shuffle(this.trainingSet, this.random);

			for (Instance i : trainingSet)
			{
				ArrayList<Double> attributes = i.attributes;
				ArrayList<Integer> classList = i.classValues;
				forwardPass(attributes);
				backwardPass(classList);
			}
			double updateVal = trainingSet.stream().mapToDouble(this::loss).average().orElse(Double.NaN);
			
			System.out.print("Epoch: " + epochCount + ", Loss: ");
			System.out.format("%.8e", updateVal);
			System.out.println("");
			
			epochCount++;
		}
	}

	public void forwardPass(ArrayList<Double> a) {
		double sum = 0.0;

		for (int i = 0; i < a.size(); i++) 
		{
			this.inputNodes.get(i).setInput(a.get(i));
		}

		for (int j = 0; j < hiddenNodes.size(); j++) 
		{
			hiddenNodes.get(j).calculateOutput();
		}

		for (int k = 0; k < outputNodes.size(); k++)
		{
			outputNodes.get(k).calculateOutput();
			sum = outputNodes.stream().mapToDouble(Node::getOutput).sum();
		}

		for (int k = 0; k < outputNodes.size(); k++)
		{
			outputNodes.get(k).setNormalOutput(sum);
		}
	}

	public void backwardPass(ArrayList<Integer> b) {
		for (int i = 0; i < b.size(); i++) 
		{
			outputNodes.get(i).setGradient(b.get(i) - outputNodes.get(i).getOutput());
		}

		for (int k = 0; k < outputNodes.size(); k++) 
		{
			outputNodes.get(k).calculateDelta();
		}

		for (int k = 0; k < outputNodes.size(); k++)
		{
			outputNodes.get(k).backPropAlg();
		}

		for (int i = 0; i < hiddenNodes.size(); i++) 
		{
			hiddenNodes.get(i).calculateDelta();
		}

		for (int i = 0; i < outputNodes.size(); i++) 
		{
			outputNodes.get(i).updateWeight(this.learningRate);
		}

		for (int i = 0; i < hiddenNodes.size(); i++)
		{
			hiddenNodes.get(i).updateWeight(this.learningRate);
		}
	}

	/**
	 * Calculate the cross entropy loss from the neural network for a single
	 * instance. The parameter is a single instance
	 */
	private double loss(Instance instance) {
		ArrayList<Double> attrList = instance.attributes;
		ArrayList<Integer> classList = instance.classValues;

		forwardPass(attrList);

		double newOut = outputNodes.get(classList.indexOf(1)).getOutput();

		return -Math.log(newOut);
	}
}