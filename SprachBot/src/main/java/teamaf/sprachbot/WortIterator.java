package teamaf.sprachbot;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class WortIterator implements DataSetIterator
{

    private int charIndex = 0;
    private HashMap<Character, Integer> charMap = new HashMap<>();
    private ArrayList<Character> labels;
    private ArrayList<String> words;
    private int currentWord = 0;

    public WortIterator(String[] wordinput)
    {
        addCharacterEncoding();
        words = new ArrayList<>(wordinput.length);
        labels = new ArrayList<>(wordinput.length);
        for (String word : wordinput)
        {
            labels.add(word.charAt(0));
            words.add(word.substring(1));
        }
    }

    private void addCharacterEncoding()
    {
        for (char c : "MFNABCDEGHIJKLOPQRSTUVWXYZÄÖÜß ".toCharArray())
        {
            if (!charMap.containsKey(c))
            {
                charMap.put(c, charIndex++);
            }
        }
    }

    @Override
    public DataSet next(int i)
    {
        INDArray inputs = Nd4j.zeros(new int[]
        {
            i, charMap.size() * Main.numchars
        });
        INDArray lbls = Nd4j.zeros(new int[]
        {
            i, Main.genders
        });
        for (int k = 0; k < i; k++)
        {
            lbls.putScalar(new int[]
            {
                k, 0
            }, -1.0);
            lbls.putScalar(new int[]
            {
                k, 1
            }, -1.0);
            lbls.putScalar(new int[]
            {
                k, 2
            }, -1.0);
            lbls.putScalar(new int[]
            {
                k, charMap.get(labels.get(currentWord))
            }, 1.0);
            for (int j = 0; j < Main.numchars; j++)
            {
                inputs.putScalar(new int[]
                {
                    k, charMap.get(words.get(currentWord).charAt(j)) + j * charMap.size()
                }, 1.0);
            }
            currentWord++;
            if (!hasNext())
            {
                reset();
            }
        }
        return new DataSet(inputs, lbls);
    }

    @Override
    public int totalExamples()
    {
        return words.size();
    }

    @Override
    public int inputColumns()
    {
        return charMap.size() * Main.numchars;
    }

    @Override
    public int totalOutcomes()
    {
        return Main.genders;
    }

    @Override
    public boolean resetSupported()
    {
        return true;
    }

    @Override
    public boolean asyncSupported()
    {
        return false;
    }

    @Override
    public void reset()
    {
        currentWord = 0;
    }

    @Override
    public int batch()
    {
        return 1;
    }

    @Override
    public int cursor()
    {
        return currentWord;
    }

    @Override
    public int numExamples()
    {
        return totalExamples();
    }

    @Override
    public List<String> getLabels()
    {
        ArrayList<String> temp = new ArrayList<>(words.size());
        for (char l : labels)
        {
            temp.add(String.format("%c", l));
        }
        return temp;
    }

    @Override
    public boolean hasNext()
    {
        return currentWord >= 0 && currentWord < totalExamples() - 1;
    }

    @Override
    public DataSet next()
    {
        INDArray inputs = Nd4j.zeros(new int[]
        {
            1, charMap.size() * Main.numchars
        }, 'f');
        INDArray lbls = Nd4j.zeros(new int[]
        {
            1, Main.genders
        }, 'f');
        lbls.putScalar(new int[]
        {
            0, 0
        }, -1.0);
        lbls.putScalar(new int[]
        {
            0, 1
        }, -1.0);
        lbls.putScalar(new int[]
        {
            0, 2
        }, -1.0);
        lbls.putScalar(new int[]
        {
            0, charMap.get(labels.get(currentWord))
        }, 1.0);
        for (int i = 0; i < Main.numchars; i++)
        {
            inputs.putScalar(new int[]
            {
                0, charMap.get(words.get(currentWord).charAt(i)) + charMap.size() * i
            }, 1.0);
        }
        currentWord++;
        if (!hasNext())
        {
            reset();
        }
        return new DataSet(inputs, lbls);
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dspp)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public DataSetPreProcessor getPreProcessor()
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public HashMap<Character, Integer> charMap()
    {
        return charMap;
    }
}
