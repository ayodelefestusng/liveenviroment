// static/js/chat.js
class ChatWidget {
    constructor() {
        this.isOpen = false;
        this.isMinimized = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.setupAccessibility();
    }

    bindEvents() {
        // Toggle chat
        document.querySelector('.chat-toggle').addEventListener('click', () => {
            this.toggleChat();
        });

        // Minimize chat
        document.querySelector('.chat-minimize').addEventListener('click', (e) => {
            e.stopPropagation();
            this.minimizeChat();
        });

        // Close chat
        document.querySelector('.chat-close').addEventListener('click', (e) => {
            e.stopPropagation();
            this.closeChat();
        });

        // Send message
        document.querySelector('.chat-send-btn').addEventListener('click', () => {
            this.sendMessage();
        });

        // Send message on Enter key
        document.querySelector('.chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.closeChat();
            }
        });

        // Click outside to close
        document.addEventListener('click', (e) => {
            if (this.isOpen && !e.target.closest('.chat-widget')) {
                this.closeChat();
            }
        });

        // Handle minimized header click
        document.querySelector('.chat-header').addEventListener('click', () => {
            if (this.isMinimized) {
                this.expandChat();
            }
        });
    }

    setupAccessibility() {
        const toggle = document.querySelector('.chat-toggle');
        const container = document.querySelector('.chat-container');

        toggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.toggleChat();
            }
        });

        // Trap focus within chat when open
        container.addEventListener('keydown', (e) => {
            if (e.key === 'Tab' && this.isOpen) {
                this.trapFocus(e);
            }
        });
    }

    trapFocus(e) {
        const focusableElements = this.getFocusableElements();
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey) {
            if (document.activeElement === firstElement) {
                lastElement.focus();
                e.preventDefault();
            }
        } else {
            if (document.activeElement === lastElement) {
                firstElement.focus();
                e.preventDefault();
            }
        }
    }

    getFocusableElements() {
        const container = document.querySelector('.chat-container');
        return container.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
    }

    toggleChat() {
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        const container = document.querySelector('.chat-container');
        const toggle = document.querySelector('.chat-toggle');

        container.classList.add('active');
        toggle.classList.add('active');
        toggle.setAttribute('aria-expanded', 'true');
        container.setAttribute('aria-hidden', 'false');

        this.isOpen = true;
        this.isMinimized = false;

        // Focus on input when opening
        setTimeout(() => {
            document.querySelector('.chat-input').focus();
        }, 300);

        // Hide notification
        this.hideNotification();

        // Track analytics
        this.trackEvent('chat_opened');
    }

    closeChat() {
        const container = document.querySelector('.chat-container');
        const toggle = document.querySelector('.chat-toggle');

        container.classList.remove('active');
        toggle.classList.remove('active');
        toggle.setAttribute('aria-expanded', 'false');
        container.setAttribute('aria-hidden', 'true');

        this.isOpen = false;
        this.isMinimized = false;

        // Track analytics
        this.trackEvent('chat_closed');
    }

    minimizeChat() {
        const container = document.querySelector('.chat-container');
        
        container.classList.add('minimized');
        this.isMinimized = true;

        // Track analytics
        this.trackEvent('chat_minimized');
    }

    expandChat() {
        const container = document.querySelector('.chat-container');
        
        container.classList.remove('minimized');
        this.isMinimized = false;

        // Focus on input
        document.querySelector('.chat-input').focus();

        // Track analytics
        this.trackEvent('chat_expanded');
    }

    sendMessage() {
        const input = document.querySelector('.chat-input');
        const message = input.value.trim();

        if (message) {
            this.addMessage(message, 'user');
            input.value = '';

            // Show typing indicator
            this.showTypingIndicator();

            // Simulate response (replace with actual chat service)
            setTimeout(() => {
                this.hideTypingIndicator();
                this.generateResponse(message);
            }, 1500);

            // Track analytics
            this.trackEvent('message_sent');
        }
    }

    addMessage(text, sender) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${sender}-message`;
        messageElement.innerHTML = `
            <div class="message-bubble ${sender}-bubble">
                <p>${this.escapeHtml(text)}</p>
                <span class="message-time">${this.getCurrentTime()}</span>
            </div>
        `;

        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showTypingIndicator() {
        document.querySelector('.typing-indicator').style.display = 'flex';
        document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
    }

    hideTypingIndicator() {
        document.querySelector('.typing-indicator').style.display = 'none';
    }

    generateResponse(userMessage) {
        const responses = {
            demo: "Great! You can book a personalized demo through our website. Would you like me to direct you to the booking page?",
            pricing: "Our pricing is customized based on your specific needs. Let me connect you with our sales team for a detailed quote.",
            contact: "You can reach our team at info@dignityconcept.com or call +234 123 456 789. We're here to help!",
            default: "Thank you for your message! Our team will get back to you shortly. In the meantime, you might find answers in our documentation or feel free to ask another question."
        };

        const lowerMessage = userMessage.toLowerCase();
        let response = responses.default;

        if (lowerMessage.includes('demo') || lowerMessage.includes('meeting')) {
            response = responses.demo;
        } else if (lowerMessage.includes('price') || lowerMessage.includes('cost')) {
            response = responses.pricing;
        } else if (lowerMessage.includes('contact') || lowerMessage.includes('call')) {
            response = responses.contact;
        }

        this.addMessage(response, 'assistant');
    }

    showNotification() {
        const notification = document.querySelector('.chat-notification');
        notification.style.display = 'flex';
    }

    hideNotification() {
        const notification = document.querySelector('.chat-notification');
        notification.style.display = 'none';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    trackEvent(eventName) {
        // Integrate with your analytics service
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, {
                'event_category': 'chat_widget'
            });
        }
    }
}

// Initialize chat widget when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.chatWidget = new ChatWidget();

    // Auto-open chat after 30 seconds if user is still on page
    setTimeout(() => {
        if (!window.chatWidget.isOpen && document.visibilityState === 'visible') {
            // Only show notification, don't auto-open
            window.chatWidget.showNotification();
        }
    }, 30000);
});

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible' && window.chatWidget) {
        window.chatWidget.showNotification();
    }
});