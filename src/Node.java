import java.util.*;

/**
 * Class for internal organization of a Neural Network. There are 5 types of
 * nodes. Check the type attribute of the node for details. Feel free to modify
 * the provided function signatures to fit your own implementation
 */

public class Node {
	private int type = 0; // 0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents = null; // Array List that will
														// contain the parents
														// (including the bias
														// node) with weights if
														// applicable

	private double inputValue = 0.0;
	private double outputValue = 0.0;
	private double outputGradient = 0.0;
	private double delta = 0.0; // input gradient

	// Create a node with a specific type
	Node(int type) {
		if (type > 4 || type < 0) {
			System.out.println("Incorrect value for node type");
			System.exit(1);
		} else
			this.type = type;

		if (type == 2 || type == 4)
			parents = new ArrayList<>();
	}

	// For an input node sets the input value which will be the value of a
	// particular attribute
	public void setInput(double inputValue) {
		if (type == 0)
			this.inputValue = inputValue;
	}

	/**
	 * Calculate the output of a node. You can get this value by using
	 * getOutput()
	 */
	public void calculateOutput() {

		// Not an input or bias node
		if (type == 2 || type == 4) {
			this.inputValue = 0.0;

			for (int i = 0; i < this.parents.size(); i++)
				inputValue += this.parents.get(i).weight * this.parents.get(i).node.getOutput();

			if (type == 2)
				this.outputValue = Math.max(0, this.inputValue);
			else if (type == 4)
				this.outputValue = Math.exp(this.inputValue);
		}
	}

	// Gets the output value
	public double getOutput() {

		// Input node
		if (type == 0)
			return inputValue;
		// Bias node
		else if (type == 1 || type == 3)
			return 1.00;
		else
			return outputValue;
	}

	public void calculateDelta() {
		if (type == 2 || type == 4) {
			if (type == 2) {
				if (this.inputValue > 0)
					this.delta = this.outputGradient;
				else
					this.delta = 0;
			}
			if (type == 4)
				this.delta = this.outputGradient;
			this.outputGradient = 0.0;
		}
	}

	public void updateWeight(double learningRate) {
		if (type == 2 || type == 4) {
			for (int i = 0; i < parents.size(); i++)
				parents.get(i).weight += learningRate * parents.get(i).node.getOutput() * this.delta;
		}
	}

	public void backPropAlg() {
		if (this.type == 4) {
			for (int i = 0; i < this.parents.size(); i++)
				this.parents.get(i).node.setGradient(this.parents.get(i).weight * this.delta);
		}
	}

	public double getDelta() {
		return this.delta;
	}

	public void setGradient(double newGrad) {
		if (this.type == 2 || this.type == 4)
			this.outputGradient += newGrad;
	}

	public void setNormalOutput(double newValue) {
		if (this.type == 4)
			this.outputValue /= newValue;
	}
}