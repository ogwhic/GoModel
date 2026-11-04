package mistral

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/enterpilot/gomodel/internal/core"
	"github.com/enterpilot/gomodel/internal/llmclient"
	"github.com/enterpilot/gomodel/internal/providers/providertest"
)

// modelsFixture mirrors Mistral's /v1/models listing: entries carry a
// capabilities object reporting what each model supports, and chat models
// report a context window.
const modelsFixture = `{
	"object": "list",
	"data": [
		{"id": "mistral-large-latest", "object": "model", "created": 1737360000, "owned_by": "mistralai",
			"max_context_length": 131072,
			"capabilities": {"completion_chat": true, "completion_fim": false, "function_calling": true, "vision": false, "json_mode": true}},
		{"id": "pixtral-12b-2409", "object": "model", "created": 1728000000, "owned_by": "mistralai",
			"max_context_length": 131072,
			"capabilities": {"completion_chat": true, "function_calling": true, "vision": true, "json_mode": true}},
		{"id": "codestral-latest", "object": "model", "created": 1738000000, "owned_by": "mistralai",
			"capabilities": {"completion_chat": true, "completion_fim": true, "function_calling": true, "vision": false}},
		{"id": "mistral-embed", "object": "model", "created": 1710000000, "owned_by": "mistralai",
			"capabilities": {"completion_chat": false, "function_calling": false, "vision": false}},
		{"id": "   ", "object": "model"},
		{"id": "untyped-model"}
	]
}`

func listModels(t *testing.T, body string) map[string]core.Model {
	t.Helper()
	server, capture := providertest.JSONServer(t, http.StatusOK, body)
	provider := newTestProvider("mistral-key", server.URL, server.Client(), llmclient.Hooks{})

	resp, err := provider.ListModels(context.Background())
	require.NoError(t, err)
	assert.Equal(t, "list", resp.Object)

	req := capture.Last(t)
	assert.Equal(t, http.MethodGet, req.Method)
	assert.Equal(t, "/models", req.Path)
	assert.Equal(t, "Bearer mistral-key", req.Header.Get("Authorization"))

	byID := make(map[string]core.Model, len(resp.Data))
	for _, model := range resp.Data {
		byID[model.ID] = model
	}
	return byID
}

func TestListModels_ReportsCapabilitiesAsListed(t *testing.T) {
	models := listModels(t, modelsFixture)

	assert.Len(t, models, 5, "the blank-id entry is skipped")

	large := models["mistral-large-latest"]
	require.NotNil(t, large.Metadata)
	assert.Equal(t, []string{"chat", "responses"}, large.Metadata.Modes)
	assert.Equal(t, []core.ModelCategory{core.CategoryTextGeneration}, large.Metadata.Categories)
	assert.Equal(t, map[string]bool{
		"function_calling": true,
		"vision":           false,
		"json_mode":        true,
		"completion_fim":   false,
	}, large.Metadata.Capabilities)
	require.NotNil(t, large.Metadata.ContextWindow)
	assert.Equal(t, 131072, *large.Metadata.ContextWindow)
	assert.Equal(t, "mistralai", large.OwnedBy)

	pixtral := models["pixtral-12b-2409"]
	require.NotNil(t, pixtral.Metadata)
	assert.True(t, pixtral.Metadata.Capabilities["vision"], "pixtral reports vision support")

	codestral := models["codestral-latest"]
	require.NotNil(t, codestral.Metadata)
	assert.True(t, codestral.Metadata.Capabilities["completion_fim"], "FIM support is reported under its own key")
	assert.NotContains(t, codestral.Metadata.Capabilities, "json_mode")

	embed := models["mistral-embed"]
	require.NotNil(t, embed.Metadata)
	assert.Empty(t, embed.Metadata.Modes, "no chat mode without completion_chat; nothing is guessed")
	assert.Equal(t, map[string]bool{
		"function_calling": false,
		"vision":           false,
	}, embed.Metadata.Capabilities)

	untyped := models["untyped-model"]
	require.NotNil(t, untyped.Metadata)
	assert.Empty(t, untyped.Metadata.Modes)
	assert.Empty(t, untyped.Metadata.Capabilities)
	assert.Nil(t, untyped.Metadata.ContextWindow)
}

func TestListModels_UnsupportedListing(t *testing.T) {
	server, _ := providertest.JSONServer(t, http.StatusNotFound, `{"error":{"message":"not found"}}`)
	provider := newTestProvider("mistral-key", server.URL, server.Client(), llmclient.Hooks{})

	_, err := provider.ListModels(context.Background())
	require.Error(t, err)
	assert.ErrorIs(t, err, core.ErrModelListingUnsupported, "a 404 /models listing is marked unsupported")
}
