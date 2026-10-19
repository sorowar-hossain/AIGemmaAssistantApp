using AIGemmaAssistantApp.Models;
using Microsoft.ML.OnnxRuntimeGenAI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AIGemmaAssistantApp.Services
{
    public class GemmaChatService : IDisposable
    {
        /*
        Model: loads your Gemma model.
        Tokenizer: converts text into tokens and decodes generated tokens.
        chatTemplate: loads the template that formats messages for Gemma.
        IDisposable: ensures model resources can be released when the service is no longer needed.
         */
        private readonly Model model;
        private readonly Tokenizer tokenizer;
        private readonly string chatTemplate;

        private const int MaxInputTokens = 3000;
        private const int MaxNewTokens = 256;
        public GemmaChatService(string modelPath)
        {
            model = new Model(modelPath);

            try
            {
                tokenizer = new Tokenizer(model);

                string templatePath = Path.Combine(
                    modelPath,
                    "chat_template.jinja");

                chatTemplate = File.ReadAllText(templatePath);
            }
            catch
            {
                model.Dispose();
                throw;
            }

        }


        /*
         - GetResponse() now handles prompt formatting and model inference.
            - The model and tokenizer remain loaded between calls.
            - The method returns the generated answer as a string.
            - Prompt length is checked against your 3,000-token limit.
            - Output generation is capped at 256 tokens.
            - Streaming output remains visible in the console for now.
             One detail to remember: the 3,000-token check is a safety limit for your application, 
            not the full context capacity of the model. 
            We'll add automatic removal of older conversation turns in a later step.
         */
        public string GetResponse(List<ChatMessage> messages)
        {
            if (messages == null || messages.Count == 0)
                throw new ArgumentException(
                    "Conversation messages cannot be empty.",
                    nameof(messages));


            string jsonMessages = JsonSerializer.Serialize(
                messages.Select(m => new
                {
                    role = m.Role,
                    content = m.Content
                }));

            string formattedPrompt = tokenizer.ApplyChatTemplate(
                chatTemplate,
                jsonMessages,
                "",
                true);


            var inputSequences = tokenizer.Encode(formattedPrompt);

            using var generatorParams = new GeneratorParams(model);

            // Maximum total sequence length, not output-only length.
            generatorParams.SetSearchOption("max_length", 4096);
            generatorParams.SetSearchOption("temperature", 0.7);
            generatorParams.SetSearchOption("top_p", 0.9);

            using var generator = new Generator(model, generatorParams);

            generator.AppendTokenSequences(inputSequences);

            int promptTokens = checked((int)generator.TokenCount());

            Console.WriteLine($"Prompt tokens: {promptTokens}");

            if (promptTokens > MaxInputTokens)
            {
                throw new InvalidOperationException(
                    $"The prompt contains {promptTokens} tokens. " +
                    $"The limit is {MaxInputTokens}. " +
                    "Remove older messages or shorten the input.");
            }

            using var tokenizerStream = tokenizer.CreateStream();

            var answer = new StringBuilder();
            int generatedTokenCount = 0;

           
            while (!generator.IsDone() && generatedTokenCount < MaxNewTokens)
            {
                generator.GenerateNextToken();

                var outputTokens = generator.GetNextTokens();

                foreach (var token in outputTokens)
                {
                    string piece = tokenizerStream.Decode(token);
                    answer.Append(piece);

                    generatedTokenCount++;
                   
                    if (generatedTokenCount >= MaxNewTokens)
                    {
                        break;
                    }
                }

                //Console.WriteLine($"Newly generated tokens: {generatedTokenCount}");
            }
            return answer.ToString();

        }


        // this method ensure model resoureces can be released when the service no longer needed 
        public void Dispose()
        {
            model.Dispose();
            tokenizer.Dispose();
        }

    }
}
