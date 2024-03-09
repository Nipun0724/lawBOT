// function sendMessage() {
//     var userInput = document.getElementById('user-input').value;
//     var chatBox = document.getElementById('chat-box');

//     // Display user message
//     var userMessage = document.createElement('div');
//     userMessage.className = 'message user-message';
//     userMessage.innerHTML = userInput;
//     chatBox.appendChild(userMessage);

//     // Send user input to backend server
//     fetch('/chatbot', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/x-www-form-urlencoded',
//         },
//         body: 'user_input=' + encodeURIComponent(userInput)
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Display bot response
//         var botMessage = document.createElement('div');
//         botMessage.className = 'message bot-message';
//         botMessage.innerHTML = data.bot_response;
//         chatBox.appendChild(botMessage);

//         // Clear user input
//         document.getElementById('user-input').value = '';

//         // Scroll to bottom of chat box
//         chatBox.scrollTop = chatBox.scrollHeight;
//     });
// }
