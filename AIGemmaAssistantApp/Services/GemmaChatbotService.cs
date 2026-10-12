using Microsoft.Extensions.AI;
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
            try
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

                //Console.WriteLine($"Template path: {templatePath}");

                string chatTemplate = File.ReadAllText(templatePath);

                // Store conversation history.
                var history = new List<ChatMessage>();

                Console.WriteLine("Gemma is ready!");
                Console.WriteLine("Type 'exit' to quit.");
                Console.WriteLine("Type 'clear' to clear history.");
                Console.WriteLine("Type 'help' to see available commands.");
                Console.WriteLine();



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

                // Interactive chatbot loop
                while (true)
                {
                    Console.Write("You: ");
                    string? userInput = Console.ReadLine();

                    // Handle end-of-input (for example, Ctrl+Z then Enter on Windows).
                    // This condition handles a special situation: when the console has no more input to read.
                    if (userInput is null)
                    {
                        Console.WriteLine("\nChatbot closed.");
                        break;
                    }


                    if (userInput.Equals("help", StringComparison.OrdinalIgnoreCase))
                    {
                        Console.WriteLine();
                        Console.WriteLine("========== Gemma AI Assistant ==========");
                        Console.WriteLine("Ask anything       - Send a question to Gemma");
                        Console.WriteLine("clear              - Clear conversation history");
                        Console.WriteLine("help               - Show available commands");
                        Console.WriteLine("exit               - Close the chatbot");
                        Console.WriteLine("========================================");
                        Console.WriteLine();

                        continue;
                    }


                    if (userInput.Equals("clear", StringComparison.OrdinalIgnoreCase))
                    {
                        history.Clear();

                        Console.WriteLine("Conversation history cleared!");
                        Console.WriteLine();

                        continue;
                    }

                    if (string.IsNullOrWhiteSpace(userInput))
                    {
                        Console.WriteLine("Please enter a question.");
                        continue;
                    }


                    if (userInput.Equals("exit",StringComparison.OrdinalIgnoreCase))
                    {
                        break;
                    }

                  

                    string messagesJson = JsonSerializer.Serialize(history);

                    string formattedPrompt = tokenizer.ApplyChatTemplate(
                        chatTemplate,
                        messagesJson,
                        "",
                        true);

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

                    var answer = new StringBuilder();

                    Console.WriteLine();
                    Console.Write("Gemma: ");

                    // Generate response
                    while (!generator.IsDone())
                    {
                        generator.GenerateNextToken();

                        var outputTokens = generator.GetNextTokens();

                        foreach (var token in outputTokens)
                        {
                            string piece = tokenizerStream.Decode(token);

                            Console.Write(piece);
                            answer.Append(piece);
                        }
                    }

                    Console.WriteLine();
                    Console.WriteLine();

                    // Add the user's message to history.
                    history.Add(new ChatMessage("user", userInput));
                    // Save Gemma's answer for future questions.
                    history.Add(new ChatMessage("assistant", answer.ToString()));
                }

                Console.WriteLine("Chatbot closed.");
            }

            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine($"Error generating response: {ex.Message}");
                Console.WriteLine("Please try another question.");
                Console.WriteLine();
            }
        }

        /*
            In your Gemma chatbot, this line defines a C# record named ChatMessage. 
                It represents one message in the conversation, including who sent it and what they said.
            public record ChatMessage(string role, string content);


            1. What do role and content mean?
            - role — identifies who sent the message, such as "user" or "assistant".
            - content — contains the actual message text.
         */
        public record ChatMessage(string role, string content);


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
