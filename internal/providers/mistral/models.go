package mistral

import (
	"context"
	"net/http"
	"strings"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/llmclient"
	"github.com/enterpilot/gomodel/internal/providers"
)

// modelsResponse mirrors Mistral's /v1/models payload. Entries carry the
// OpenAI-compatible fields plus a capabilities object in which Mistral
// reports, per model, what the model supports (completion_chat,
// function_calling, vision, json_mode, completion_fim, ocr, ...).
type modelsResponse struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

type modelInfo struct {
	ID               string          `json:"id"`
	Object           string          `json:"object"`
	Created          int64           `json:"created"`
	OwnedBy          string          `json:"owned_by"`
	MaxContextLength *int            `json:"max_context_length"`
	Capabilities     map[string]bool `json:"capabilities"`
}

// ListModels returns Mistral's model listing with the capabilities each
// entry reports, recorded under the catalog's capability keys — including
// the ones reported as false, so a model Mistral lists without vision is
// not advertised as vision-capable here. Models with completion_chat get the
// chat and responses modes; the listing says nothing about the others, so no
// mode is guessed for them.
func (p *Provider) ListModels(ctx context.Context) (*core.ModelsResponse, error) {
	var upstream modelsResponse
	if err := p.Do(ctx, llmclient.Request{
		Method:   http.MethodGet,
		Endpoint: "/models",
	}, &upstream); err != nil {
		return nil, core.MarkModelListingUnsupported(err)
	}

	resp := &core.ModelsResponse{Object: upstream.Object, Data: make([]core.Model, 0, len(upstream.Data))}
	for _, model := range upstream.Data {
		if strings.TrimSpace(model.ID) == "" {
			continue
		}
		resp.Data = append(resp.Data, model.toCore())
	}
	if strings.TrimSpace(resp.Object) == "" {
		resp.Object = "list"
	}
	return resp, nil
}

func (m modelInfo) toCore() core.Model {
	object := strings.TrimSpace(m.Object)
	if object == "" {
		object = "model"
	}
	return core.Model{
		ID:       strings.TrimSpace(m.ID),
		Object:   object,
		OwnedBy:  strings.TrimSpace(m.OwnedBy),
		Created:  m.Created,
		Metadata: m.metadata(),
	}
}

func (m modelInfo) metadata() *core.ModelMetadata {
	metadata := &core.ModelMetadata{}
	var capabilities map[string]bool
	for name, supported := range m.Capabilities {
		if name == "completion_chat" {
			continue
		}
		capabilities = providers.SetCapability(capabilities, name, supported)
	}
	if chat := m.Capabilities["completion_chat"]; chat {
		metadata.Modes = []string{"chat", "responses"}
		metadata.Categories = core.CategoriesForModes(metadata.Modes)
	}
	if len(capabilities) > 0 {
		metadata.Capabilities = capabilities
	}
	if m.MaxContextLength != nil && *m.MaxContextLength > 0 {
		metadata.ContextWindow = new(*m.MaxContextLength)
	}
	return metadata
}
