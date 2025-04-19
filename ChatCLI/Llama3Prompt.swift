import Foundation

/// Builds a Meta‑Llama‑3 **Instruct** prompt following
/// https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
struct Llama3Prompt {
    static func build(_ history: [ChatMessage]) -> String {
        var s = "<|begin_of_text|>"
        for (index, msg) in history.enumerated() {
            switch msg.role {
            case .system where index == 0:
                s += "<|start_header_id|>system<|end_header_id|>\n"
                s += msg.content
                s += "<|eot_id|>"
            case .user:
                s += "<|start_header_id|>user<|end_header_id|>\n"
                s += msg.content
                s += "<|eot_id|>"
            case .assistant:
                s += "<|start_header_id|>assistant<|end_header_id|>\n"
                s += msg.content
                s += "<|eot_id|>"
            default:
                break
            }
        }

        // If the last turn is from the user, append the assistant header so the
        // model knows where to continue generation.
        if history.last?.role == .user {
            s += "<|start_header_id|>assistant<|end_header_id|>\n"
        }
        return s
    }
}