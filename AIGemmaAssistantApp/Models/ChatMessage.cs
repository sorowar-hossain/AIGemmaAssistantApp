using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIGemmaAssistantApp.Models
{
    public class ChatMessage
    {
        public string Role { get; set; } = string.Empty;

        public string Content { get; set; } = string.Empty;

        public ChatMessage()
        {
        }

        // The message model represents a conversation message, both user and assistant
        public ChatMessage(string role, string content)
        {
            Role = role;
            Content = content;
        }
    }
}
