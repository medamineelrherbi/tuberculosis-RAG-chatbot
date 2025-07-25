class Chatbox {
    constructor() {
        this.args = {
            // Le 'openButton' a été supprimé
            chatBox: document.querySelector('#my-chatbox'),
            sendButton: document.querySelector('.send__button'),
        };

        // Cible l'élément interne pour les interactions de message
        this.supportBox = this.args.chatBox.querySelector('.chatbox__support');
        
        this.messages = [];
    }

    display() {
        // La logique du bouton d'ouverture a été supprimée
        const { sendButton } = this.args;

        sendButton.addEventListener('click', () => this.onSendButton());

        const inputField = this.supportBox.querySelector('input');
        inputField.addEventListener('keyup', ({ key }) => {
            if (key === 'Enter') {
                this.onSendButton();
            }
        });
    }

    // La fonction 'toggleState' a été supprimée car elle n'est plus utile

    onSendButton() {
        const inputField = this.supportBox.querySelector('input');
        const text = inputField.value;

        if (text === '') {
            return;
        }

        this.addMessage({ name: 'User', message: text });

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text }),
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            this.addMessage({ name: 'Sam', message: data.answer });
            this.updateChatText();
            inputField.value = ''; // Vide le champ de saisie
        })
        .catch(error => {
            console.error('Error:', error);
            this.addMessage({ name: 'Sam', message: 'Sorry, something went wrong. Please try again later.' });
            this.updateChatText();
        });
    }

    addMessage(message) {
        this.messages.push(message);
        this.updateChatText();
    }

    updateChatText() {
        let html = '';

        this.messages.slice().reverse().forEach(item => {
            if (item.name === 'Sam') {
                html += `<div class="messages__item messages__item--operator">${item.message}</div>`;
            } else {
                html += `<div class="messages__item messages__item--visitor">${item.message}</div>`;
            }
        });

        const chatMessages = this.supportBox.querySelector('.chatbox__messages');
        chatMessages.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();