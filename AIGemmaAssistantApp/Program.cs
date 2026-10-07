using Microsoft.ML.OnnxRuntimeGenAI;

/*
 Console.WriteLine("Tokenizer methods:");

foreach (var method in typeof(Tokenizer).GetMethods())
{
    Console.WriteLine(method);
}

The key methods are:

Tokenizer.Encode(string)
Tokenizer.Decode(ReadOnlySpan<int>)
Tokenizer.ApplyChatTemplate(...)
Generator.AppendTokens(...)
Generator.GenerateNextToken()
Generator.GetNextTokens()
 
 */


string modelPath = @"C:\AIModels\Gemma2-2B-INT4";

using var model = new Model(modelPath);
using var tokenizer = new Tokenizer(model);

string prompt = "What is C#?";

string formattedPrompt = prompt;

Console.WriteLine("Formatted prompt:");
Console.WriteLine(formattedPrompt);
Console.WriteLine();

var sequences = tokenizer.Encode(formattedPrompt);

using var generatorParams = new GeneratorParams(model);

generatorParams.SetSearchOption("max_length", 4096);

using var generator = new Generator(model, generatorParams);

generator.AppendTokenSequences(sequences);

Console.WriteLine("Gemma:");

while (!generator.IsDone())
{
    generator.GenerateNextToken();

    var outputTokens = generator.GetNextTokens();

    if (outputTokens.Length > 0)
    {
        string output = tokenizer.Decode(outputTokens);
        Console.Write(output);
    }
}