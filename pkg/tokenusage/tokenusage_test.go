// Copyright (c) 2022 Alibaba Group Holding Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tokenusage

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/higress-group/wasm-go/pkg/wrapper"
)

func TestGetTokenUsage(t *testing.T) {
	for _, tc := range []struct {
		name         string
		body         string
		requestModel string
		want         TokenUsage
	}{
		{
			name: "legacy-qwen-total",
			body: `{"model":"qwen-model","output":{"embeddings":[]},"usage":{"input_tokens":3,"image_tokens":7,"total_tokens":10}}`,
			want: TokenUsage{InputToken: 10, TotalToken: 10, Model: "qwen-model", InputTokenDetails: map[string]int64{"text_tokens": 3, "image_tokens": 7}},
		},
		{
			name:         "legacy-qwen-without-total-request-model",
			body:         `{"output":{"embeddings":[]},"usage":{"input_tokens":3,"image_tokens":7,"output_tokens":2}}`,
			requestModel: "request-model",
			want:         TokenUsage{InputToken: 10, OutputToken: 2, TotalToken: 12, Model: "request-model", InputTokenDetails: map[string]int64{"text_tokens": 3, "image_tokens": 7}},
		},
		{
			name: "legacy-qwen-explicit-zero-total",
			body: `{"output":{"embeddings":[]},"usage":{"input_tokens":3,"image_tokens":7,"total_tokens":0}}`,
			want: TokenUsage{InputToken: 10, Model: ModelUnknown, InputTokenDetails: map[string]int64{"text_tokens": 3, "image_tokens": 7}},
		},
		{
			name: "legacy-qwen-image-only",
			body: `{"output":{"embeddings":[]},"usage":{"input_tokens":0,"image_tokens":7}}`,
			want: TokenUsage{InputToken: 7, TotalToken: 7, Model: ModelUnknown, InputTokenDetails: map[string]int64{"text_tokens": 0, "image_tokens": 7}},
		},
		{
			name: "legacy-qwen-without-image-tokens",
			body: `{"output":{"embeddings":[]},"usage":{"input_tokens":3}}`,
			want: TokenUsage{InputToken: 3, TotalToken: 3, Model: ModelUnknown, InputTokenDetails: map[string]int64{"text_tokens": 3}},
		},
		{
			name: "legacy-qwen-without-text-tokens",
			body: `{"output":{"embeddings":[]},"usage":{"image_tokens":7}}`,
			want: TokenUsage{InputToken: 7, TotalToken: 7, Model: ModelUnknown, InputTokenDetails: map[string]int64{"image_tokens": 7}},
		},
		{
			name: "modern-qwen-details-do-not-double-count-top-level-image",
			body: `{"output":{"embeddings":[]},"usage":{"input_tokens":10,"image_tokens":7,"input_tokens_details":{"text_tokens":3,"image_tokens":7},"output_tokens":2,"total_tokens":12}}`,
			want: TokenUsage{InputToken: 10, OutputToken: 2, TotalToken: 12, Model: ModelUnknown, InputTokenDetails: map[string]int64{"text_tokens": 3, "image_tokens": 7}},
		},
		{
			name: "modern-qwen-empty-details-take-priority",
			body: `{"output":{"embeddings":[]},"usage":{"input_tokens":10,"image_tokens":7,"input_tokens_details":{}}}`,
			want: TokenUsage{InputToken: 10, TotalToken: 10, Model: ModelUnknown},
		},
		{
			name: "doubao-json-with-blank-line",
			body: "{\"model\":\"doubao-model\",\"data\":{\"embedding\":[0.1]},\n\n\"usage\":{\"prompt_tokens\":10,\"prompt_tokens_details\":{\"text_tokens\":3,\"image_tokens\":7},\"total_tokens\":10}}",
			want: TokenUsage{InputToken: 10, TotalToken: 10, Model: "doubao-model", InputTokenDetails: map[string]int64{"text_tokens": 3, "image_tokens": 7}},
		},
		{
			name: "openai-json-ignores-top-level-image",
			body: `{"model":"gpt-model","usage":{"prompt_tokens":3,"completion_tokens":2,"image_tokens":99,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":1}}}`,
			want: TokenUsage{InputToken: 3, OutputToken: 2, TotalToken: 5, Model: "gpt-model", InputTokenDetails: map[string]int64{"cached_tokens": 1}, OutputTokenDetails: map[string]int64{"reasoning_tokens": 1}},
		},
		{
			name: "openai-sse",
			body: "data: {\"model\":\"gpt-model\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\r\n\r\ndata: {\"model\":\"gpt-model\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\r\n\r\ndata: [DONE]\r\n\r\n",
			want: TokenUsage{InputToken: 3, OutputToken: 2, TotalToken: 5, Model: "gpt-model"},
		},
		{
			name: "gemini-json",
			body: `{"modelVersion":"gemini-model","usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":9,"cachedContentTokenCount":2,"toolUsePromptTokenCount":1,"thoughtsTokenCount":2}}`,
			want: TokenUsage{InputToken: 4, OutputToken: 3, TotalToken: 9, Model: "gemini-model", InputTokenDetails: map[string]int64{"cached_content_token_count": 2, "tool_use_prompt_token_count": 1}, OutputTokenDetails: map[string]int64{"thoughts_token_count": 2}},
		},
		{
			name: "gemini-sse",
			body: "data: {\"modelVersion\":\"gemini-model\",\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":3,\"totalTokenCount\":7}}\n\n",
			want: TokenUsage{InputToken: 4, OutputToken: 3, TotalToken: 7, Model: "gemini-model"},
		},
		{
			name: "anthropic-json-cache-tokens",
			body: `{"model":"claude-model","usage":{"input_tokens":4,"output_tokens":2,"cache_creation_input_tokens":3,"cache_read_input_tokens":1}}`,
			want: TokenUsage{InputToken: 4, OutputToken: 2, TotalToken: 10, Model: "claude-model", InputTokenDetails: map[string]int64{"cache_creation_input_tokens": 3, "cache_read_input_tokens": 1}, AnthropicCacheCreationInputToken: 3, AnthropicCacheReadInputToken: 1},
		},
		{
			name: "anthropic-sse-preserves-message-start-usage",
			body: "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-model\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n\n",
			want: TokenUsage{InputToken: 4, OutputToken: 2, TotalToken: 6, Model: "claude-model"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := wrapper.NewCommonVmCtx[struct{}]("usage-test").NewPluginContext(1).NewHttpContext(1).(wrapper.HttpContext)
			ctx.SetContext(CtxKeyRequestModel, tc.requestModel)
			body := []byte(tc.body)
			original := bytes.Clone(body)
			got := GetTokenUsage(ctx, body)
			if tc.want.InputTokenDetails == nil {
				tc.want.InputTokenDetails = map[string]int64{}
			}
			if tc.want.OutputTokenDetails == nil {
				tc.want.OutputTokenDetails = map[string]int64{}
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("GetTokenUsage() = %+v, want %+v", got, tc.want)
			}
			if !bytes.Equal(body, original) {
				t.Errorf("GetTokenUsage() changed response body: got %q, want %q", body, original)
			}
			for key, want := range map[string]any{
				CtxKeyInputToken:         got.InputToken,
				CtxKeyOutputToken:        got.OutputToken,
				CtxKeyTotalToken:         got.TotalToken,
				CtxKeyInputTokenDetails:  got.InputTokenDetails,
				CtxKeyOutputTokenDetails: got.OutputTokenDetails,
				CtxKeyModel:              got.Model,
			} {
				if attr := ctx.GetUserAttribute(key); !reflect.DeepEqual(attr, want) {
					t.Errorf("context attribute %q = %#v, want %#v", key, attr, want)
				}
			}
		})
	}
}
