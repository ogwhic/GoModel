// Package mistral provides Mistral AI (La Plateforme) integration for the LLM gateway.
package mistral

import (
	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/providers"
	"github.com/enterpilot/gomodel/internal/providers/openai"
)

const defaultBaseURL = "https://api.mistral.ai/v1"

// Registration provides factory registration for the Mistral provider.
var Registration = providers.Registration{
	Type: "mistral",
	New:  New,
	Discovery: providers.DiscoveryConfig{
		DefaultBaseURL: defaultBaseURL,
	},
}

// Provider implements Mistral's OpenAI-compatible API through the shared
// chat-centric adapter: chat completions, embeddings, and passthrough go
// upstream unchanged, and Responses requests are translated through chat
// completions. Only the model listing differs (see ListModels), because
// Mistral's /v1/models reports each model's capabilities, which the listing
// maps onto the gateway's metadata as reported instead of guessing.
type Provider struct {
	*openai.ChatCompatible
}

var (
	_ core.Provider            = (*Provider)(nil)
	_ core.PassthroughProvider = (*Provider)(nil)
)

// New creates a new Mistral provider.
func New(cfg providers.ProviderConfig, opts providers.ProviderOptions) core.Provider {
	return &Provider{openai.NewChatCompatible(cfg.APIKey, opts, openai.CompatibleProviderConfig{
		ProviderName: "mistral",
		BaseURL:      providers.ResolveBaseURL(cfg.BaseURL, defaultBaseURL),
	})}
}
