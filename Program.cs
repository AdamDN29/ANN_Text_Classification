using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DataAccess;

namespace ANN_Text_Classification
{
    class NeuralNetWork
    {
        private Random _radomObj;

        public NeuralNetWork(int synapseMatrixColumns, int synapseMatrixLines)
        {
            SynapseMatrixColumns = synapseMatrixColumns;
            SynapseMatrixLines = synapseMatrixLines;

            _Init();
        }

        public int SynapseMatrixColumns { get; }
        public int SynapseMatrixLines { get; }
        public double[,] SynapsesMatrix { get; private set; }

        /// Initialize the ramdom object and the matrix of random weights
        private void _Init()
        {
            // get random weights
            _radomObj = new Random(1);
            _GenerateSynapsesMatrix();
        }

        /// Generate our matrix with the weight of the synapses
        private void _GenerateSynapsesMatrix()
        {
            SynapsesMatrix = new double[SynapseMatrixLines, SynapseMatrixColumns];

            for (var i = 0; i < SynapseMatrixLines; i++)
            {
                for (var j = 0; j < SynapseMatrixColumns; j++)
                {
                    SynapsesMatrix[i, j] = (2 * _radomObj.NextDouble()) - 1;
                }
            }
        }

        /// Calculate the sigmoid of a value
        private double[,] _CalculateSigmoid(double[,] matrix)
        {

            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    var value = matrix[i, j];
                    matrix[i, j] = 1 / (1 + Math.Exp(value * -1));
                }
            }
            return matrix;
        }

        /// Calculate the sigmoid derivative of a value
        private double[,] _CalculateSigmoidDerivative(double[,] matrix)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    var value = matrix[i, j];
                    matrix[i, j] = value * (1 - value);
                }
            }
            return matrix;
        }

        /// Will return the outputs give the set of the inputs
        public double[,] Think(double[,] inputMatrix)
        {
            var productOfTheInputsAndWeights = MatrixDotProduct(inputMatrix, SynapsesMatrix);

            return _CalculateSigmoid(productOfTheInputsAndWeights);

        }

        /// Train the neural network to achieve the output matrix values
        public void Train(double[,] trainInputMatrix, double[,] trainOutputMatrix, int interactions)
        {
            // we run all the interactions
            for (var i = 0; i < interactions; i++)
            {
                // calculate the output
                var output = Think(trainInputMatrix);

                // calculate the error
                var error = MatrixSubstract(trainOutputMatrix, output);
                var curSigmoidDerivative = _CalculateSigmoidDerivative(output);
                var error_SigmoidDerivative = MatrixProduct(error, curSigmoidDerivative);

                // calculate the adjustment :) 
                var adjustment = MatrixDotProduct(MatrixTranspose(trainInputMatrix), error_SigmoidDerivative);

                SynapsesMatrix = MatrixSum(SynapsesMatrix, adjustment);
            }
        }

        /// Transpose a matrix
        public static double[,] MatrixTranspose(double[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            double[,] result = new double[h, w];

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }

            return result;
        }

        /// Sum one matrix with another
        public static double[,] MatrixSum(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] + matrixb[i, u];
                }
            }

            return result;
        }

        /// Subtract one matrix from another
        public static double[,] MatrixSubstract(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] - matrixb[i, u];
                }
            }

            return result;
        }

        /// Multiplication of a matrix
        public static double[,] MatrixProduct(double[,] matrixa, double[,] matrixb)
        {
            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var result = new double[rowsA, colsA];

            for (int i = 0; i < rowsA; i++)
            {
                for (int u = 0; u < colsA; u++)
                {
                    result[i, u] = matrixa[i, u] * matrixb[i, u];
                }
            }

            return result;
        }

        /// Dot Multiplication of a matrix
        public static double[,] MatrixDotProduct(double[,] matrixa, double[,] matrixb)
        {

            var rowsA = matrixa.GetLength(0);
            var colsA = matrixa.GetLength(1);

            var rowsB = matrixb.GetLength(0);
            var colsB = matrixb.GetLength(1);

            if (colsA != rowsB)
                throw new Exception("Matrices dimensions don't fit.");

            var result = new double[rowsA, colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    for (int k = 0; k < rowsB; k++)
                        result[i, j] += matrixa[i, k] * matrixb[k, j];
                }
            }
            return result;
        }

    }
    class Program
    {
        static void PrintMatrix(double[,] matrix, int flag)
        {
            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    if (flag == 1)
                    {
                        Console.Write(string.Format("{0} ", Math.Round(matrix[i, j])));
                    }
                    else if (flag == 2)
                    {
                        if (Math.Round(matrix[i, j]) == 1)
                        {
                            Console.Write("Ya");
                        }
                        else { Console.Write("Tidak"); }
                    }
                    else
                    {
                        Console.Write(string.Format("{0} ", matrix[i, j]));
                    }
                    
                }
                Console.Write(Environment.NewLine);
            }
        }

        static double[,] CreateNode(List<string> x, IReadOnlyList<string> vocabulary)
        {
            string[] wordsTemp = x[0].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            double[,] node = new double[x.Count, wordsTemp.Length];

            for (int i = 0; i < x.Count; i++)
            {
                string[] words = x[i].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);

                //var text = new double[words.Length];

                for (int j = 0; j < words.Length; j++)
                {
                    for (int k = 0; k < vocabulary.Count; k++)
                    {
                        if(words[j] == vocabulary[k])
                        {
                            node[i, j] = (k + 1) % 2;
                            //Console.WriteLine("Matrix [{0} , {1}] = {2}", i, j, node[i, j]);
                        }
       
                    }
                }   
            }
            return node;
        }

        static void Main(string[] args)
        {
            // Get Data and Vocabulary file
            const string dataFilePath = @"D:\1. KULIAH\7. Semester 7\Machine Learning\UTS\ANN\train_ann.csv";

            // Convert Data
            var dataTable = DataTable.New.ReadCsv(dataFilePath);
            List<string> x = dataTable.Rows.Select(row => row["Text"]).ToList();
            double[] y = dataTable.Rows.Select(row => double.Parse(row["IsTarget"]))
                                       .ToArray();
            // Atribute Length
            string[] atribute_data = x[0].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            var atribute_length = atribute_data.Length;


            // Vocabulary by File Vocab
            const string vocabFilePath = @"D:\1. KULIAH\7. Semester 7\Machine Learning\UTS\ANN\vocab_ann.csv";
            var vocabTable = DataTable.New.ReadCsv(vocabFilePath);
            List<string> vocabulary = vocabTable.Rows.Select(row => row["vocabulary"]).ToList();

            Console.WriteLine(new string('=', 50));
            Console.WriteLine("              Text Classification          ");
            Console.WriteLine("           Artificial Neural Network          ");
            Console.WriteLine(new string('=', 50));

            // Show Vocabulary
            Console.WriteLine(new string('=', 50));
            Console.WriteLine("Vocabulary  ");
            Console.WriteLine(new string('=', 50));
            for (int i = 0; i < vocabulary.Count; i++)
            {
                Console.WriteLine("Vocabulary {0} = {1} ({2})", i + 1, vocabulary[i], (i+1)%2 );
            }
            Console.WriteLine(new string('=', 50));

            // Show Text
            Console.WriteLine("Text ");
            Console.WriteLine(new string('=', 50));
            for (int i = 0; i < x.Count; i++)
            {
                Console.WriteLine("Text {0} = {1}", i + 1, x[i]);
            }
            Console.WriteLine(new string('=', 50));

            // Show Target
            Console.WriteLine("Target ");
            Console.WriteLine(new string('=', 50));
            for (int i = 0; i < y.Length; i++)
            {
                Console.WriteLine("Target {0} = {1}", i + 1, y[i]);
            }
            Console.WriteLine(new string('=', 50));

            // Create and Show Text in Node
            var trainingInputs = CreateNode(x, vocabulary);
            Console.WriteLine("Text in Node ");
            Console.WriteLine(new string('=', 50));
            PrintMatrix(trainingInputs, 0);
            Console.WriteLine(new string('=', 50));

            // Make Training Output Matrix
            var trainingOutputs = new double[y.Length, 1];

            for (int i = 0; i < y.Length; i++)
            {
                trainingOutputs[i, 0] = y[i];
            }

            // Set the Matrix size
            var curNeuralNetwork = new NeuralNetWork(1, atribute_length);

            Console.WriteLine("Synaptic Weights ");
            Console.WriteLine(new string('=', 50));

            // Show Synaptic weights before training
            Console.WriteLine("Synaptic weights before training:");
            PrintMatrix(curNeuralNetwork.SynapsesMatrix, 0);

            // Train the Training Data
            curNeuralNetwork.Train(trainingInputs, trainingOutputs, 10000);

            // Show Synaptic weights after training
            Console.WriteLine("\nSynaptic weights after training:");
            PrintMatrix(curNeuralNetwork.SynapsesMatrix, 0);

            // Testing neural networks by data test
            // Get Data Test
            const string testFilePath = @"D:\1. KULIAH\7. Semester 7\Machine Learning\UTS\ANN\test_ann.csv";

            var testTable = DataTable.New.ReadCsv(testFilePath);
            List<string> a_test = testTable.Rows.Select(row => row["Text"]).ToList();
            double[] b_test = testTable.Rows.Select(row => double.Parse(row["IsTarget"]))
                                       .ToArray();

            var testInputs = CreateNode(a_test, vocabulary);

            var testOutputs = new double[y.Length, 1];

            for (int i = 0; i < y.Length; i++)
            {
                testOutputs[i, 0] = y[i];
            }

            // Get Accuracy by Data Test
            int accuracy2 = 0;
            
            Console.WriteLine(new string('=', 50));
            Console.WriteLine("Akurasi Model ");

            for (int i = 0; i < a_test.Count; i++)
            {
                // Predict the input
                var temp = new double[1, atribute_length];
                for (int j = 0; j < atribute_length; j++)
                {
                    temp[0, j] = testInputs[i, j];
                }

                var test = curNeuralNetwork.Think(temp);
                //PrintMatrix(temp, 0);
                //PrintMatrix(test, 0);
                //Console.WriteLine("Target = {0}", b_test[i]);
                var result = -1;
                if (Math.Round(test[0, 0]) == 1)
                {
                    result = 1;
                }
                if (result == b_test[i])
                {
                    accuracy2 += 1;
                }
            }

            var accuracy3 = (double)(accuracy2 * 100) / a_test.Count;
            Console.WriteLine(new string('=', 50));
            Console.WriteLine("Jumlah Data Test = {0}", a_test.Count);
            Console.WriteLine("Jumlah Test benar =  {0}", accuracy2);
            Console.WriteLine("Tingkat Akurasi dengan Data Test = {0} %", accuracy3);

            Console.WriteLine(new string('=', 50));

            //Console.WriteLine("\n");
            Console.WriteLine(new string('=', 50));
            Console.WriteLine("Model telah dilatih. \r\nMasukan data untuk prediksi. \n(contoh: hujan rendah sibuk lelah)");
            Console.WriteLine(new string('=', 50));

            // Predict by user input
            string userInput;
            do
            {
                userInput = Console.ReadLine();

                // Make a matrix
                var predicted = CreateUserInputNode(userInput, vocabulary);

                // Predict the input
                var Output = curNeuralNetwork.Think(predicted);
                var result = "";

                if (Math.Round(Output[0, 0]) == 1)
                {
                    result = "Ya";
                }
                else { result = "Tidak"; ; }

                Console.WriteLine(new string('=', 50));
                Console.WriteLine("Joging = {0}", result);
                Console.WriteLine(new string('=', 50));
            } while (userInput != "quit");

            Console.WriteLine("");
        }

        static double[,] CreateUserInputNode(string x, IReadOnlyList<string> vocabulary)
        {
            string[] wordsTemp = x.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            double[,] node = new double[1, wordsTemp.Length];

            for (int i = 0; i < wordsTemp.Length; i++)
            {
                for (int k = 0; k < vocabulary.Count; k++)
                {
                    if (wordsTemp[i] == vocabulary[k])
                    {
                        node[0, i] = (k + 1) % 2;
                        //Console.WriteLine("Matrix Input [0 , {0}] = {1}", i, node[0, i]);
                    }
                }
            }
            return node;
        } 
    }
}
