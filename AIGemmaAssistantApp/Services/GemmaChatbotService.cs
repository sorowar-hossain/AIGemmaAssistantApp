using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;


namespace AIGemmaAssistantApp.Services
{
    public class GemmaChatbotService
    {
        public void StartChat()
        {

            /* Question?????????

                 If you can already chat with ChatGPT or Gemini, why build your own Gemma application?

                The answer is: you are not building it because your application will be better than ChatGPT.
                You are building it to learn how LLM applications actually work and to build AI features into your own software.

                Think about it this way

                You already know how to use SQL Server.

                But you didn't learn C# by saying:

                "Why should I write a C# application? I can use Excel."

                You learned how software works so you can build your own systems.

             */

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




            string modelPath = Path.Combine(
                AppContext.BaseDirectory,
                "AIModels",
                "Gemma2-2B-INT4");

            //string modelPath = @"C:\AIModels\Gemma2-2B-INT4";

            using var model = new Model(modelPath);
            using var tokenizer = new Tokenizer(model);

            string templatePath = Path.Combine(
                modelPath,
                "chat_template.jinja");

            Console.WriteLine($"Template path: {templatePath}");

            string chatTemplate = File.ReadAllText(templatePath);

            //  Please provide me with a question or a task you'd like me to help you with!▁▁
            //string userAsk = "What is C#?";
            //string userAsk = "how to improvement my technical skill?";
            //string userAsk = "Tell me about footballer Messi";

            //string userAsk = "How to learn english?";

            //string userAsk = "how far distance from gulshan 2 to Notunbazar in dhaka bangladesh";

            string userAsk = "how to improve my AI.Net skill day by day";

            var messages = new[]
            {
                new
                {
                    role = "user",
                    content = userAsk
                }
            };

            string messagesJson = JsonSerializer.Serialize(messages);

            Console.WriteLine("Messages JSON:");
            Console.WriteLine(messagesJson);

            string formattedPrompt = tokenizer.ApplyChatTemplate(
                chatTemplate,
                messagesJson,
                "",
                true);

            Console.WriteLine("Formatted prompt:");
            Console.WriteLine(formattedPrompt);

            /*
                That means Gemma now understands:

                <start_of_turn>user
                        ↓
                "What is C#?"
                        ↓
                <end_of_turn>
                        ↓
                <start_of_turn>model
                        ↓
                "Now generate the answer"

            *****************
                Now we need:

                formattedPrompt
                     ↓
                Tokenizer
                     ↓
                Token IDs
                     ↓
                Generator
                     ↓
                Gemma response
             */

            // Convert prompt → token IDs
            var sequences = tokenizer.Encode(formattedPrompt);

            // Configure generation
            using var generatorParams = new GeneratorParams(model);

            generatorParams.SetSearchOption("max_length", 4096);

            // Create generator
            using var generator = new Generator(model, generatorParams);

            // Give prompt tokens to Gemma
            generator.AppendTokenSequences(sequences);

            // Token stream for readable output
            using var tokenizerStream = tokenizer.CreateStream();

            Console.WriteLine("Gemma:");
            Console.WriteLine();

            // Generate response
            while (!generator.IsDone())
            {
                generator.GenerateNextToken();

                var outputTokens = generator.GetNextTokens();

                foreach (var token in outputTokens)
                {
                    Console.Write(tokenizerStream.Decode(token));
                }
            }

            Console.WriteLine();
        }

    }
}

/*
    Full Gemma Flow

        Your application is essentially doing this:

                YOUR C# APPLICATION
                       │
                       ▼
              1. User Question
                       │
                       ▼
              2. Chat Messages
                       │
                       ▼
              3. JSON Serialization
                       │
                       ▼
              4. Chat Template
                       │
                       ▼
              5. Tokenization
                       │
                       ▼
                 Token IDs
                       │
                       ▼
              6. Embeddings
                       │
                       ▼
              7. Transformer
                 ┌─────────────┐
                 │ Self-Attn   │
                 │     ↓       │
                 │ Add & Norm  │
                 │     ↓       │
                 │    FFN      │
                 │     ↓       │
                 │ Add & Norm  │
                 └─────────────┘
                       │
                       ▼
              8. Next Token Prediction
                       │
                       ▼
                 Logits/Scores
                       │
                       ▼
                  Softmax
                       │
                       ▼
              9. Select Next Token
                       │
                       ▼
             Repeat generation
                       │
                       ▼
              10. Token IDs
                       │
                       ▼
              11. Tokenizer Decode
                       │
                       ▼
                Text Response
                       │
                       ▼
                 USER SEES IT
 
 
 */
