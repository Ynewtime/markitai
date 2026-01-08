# Claude 入门指南

进行您的第一次 Claude API 调用并构建一个简单的网络搜索助手

---

## 前置条件

- 一个 Anthropic [Console 账户](/)
- 一个 [API 密钥](/settings/keys)

## 调用 API

<Tabs>
  <Tab title="cURL">
    <Steps>
      <Step title="设置您的 API 密钥">
        从 [Claude Console](/settings/keys) 获取您的 API 密钥并将其设置为环境变量：

        ```bash
        export ANTHROPIC_API_KEY='your-api-key-here'
        ```
      </Step>

      <Step title="进行您的第一次 API 调用">
        运行此命令以创建一个简单的网络搜索助手：

        ```bash
        curl https://api.anthropic.com/v1/messages \
          -H "Content-Type: application/json" \
          -H "x-api-key: $ANTHROPIC_API_KEY" \
          -H "anthropic-version: 2023-06-01" \
          -d '{
            "model": "claude-sonnet-4-5",
            "max_tokens": 1000,
            "messages": [
              {
                "role": "user", 
                "content": "What should I search for to find the latest developments in renewable energy?"
              }
            ]
          }'
        ```

        **示例输出：**
        ```json
        {
          "id": "msg_01HCDu5LRGeP2o7s2xGmxyx8",
          "type": "message", 
          "role": "assistant",
          "content": [
            {
              "type": "text",
              "text": "Here are some effective search strategies to find the latest renewable energy developments:\n\n## Search Terms to Use:\n- \"renewable energy news 2024\"\n- \"clean energy breakthrough\"\n- \"solar/wind/battery technology advances\"\n- \"green energy innovations\"\n- \"climate tech developments\"\n- \"energy storage solutions\"\n\n## Best Sources to Check:\n\n**News & Industry Sites:**\n- Renewable Energy World\n- GreenTech Media (now Wood Mackenzie)\n- Energy Storage News\n- CleanTechnica\n- PV Magazine (for solar)\n- WindPower Engineering & Development..."
            }
          ],
          "model": "claude-sonnet-4-5",
          "stop_reason": "end_turn",
          "usage": {
            "input_tokens": 21,
            "output_tokens": 305
          }
        }
        ```
      </Step>
    </Steps>
  </Tab>

  <Tab title="Python">
    <Steps>
      <Step title="设置您的 API 密钥">
        从 [Claude Console](/settings/keys) 获取您的 API 密钥并将其设置为环境变量：

        ```bash
        export ANTHROPIC_API_KEY='your-api-key-here'
        ```
      </Step>

      <Step title="安装 SDK">
        安装 Anthropic Python SDK：

        ```bash
        pip install anthropic
        ```
      </Step>

      <Step title="创建您的代码">
        将其保存为 `quickstart.py`：

        ```python
        import anthropic

        client = anthropic.Anthropic()

        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": "What should I search for to find the latest developments in renewable energy?"
                }
            ]
        )
        print(message.content)
        ```
      </Step>

      <Step title="运行您的代码">
        ```bash
        python quickstart.py
        ```

        **示例输出：**
        ```python
        [TextBlock(text='Here are some effective search strategies for finding the latest renewable energy developments:\n\n**Search Terms to Use:**\n- "renewable energy news 2024"\n- "clean energy breakthroughs"\n- "solar/wind/battery technology advances"\n- "energy storage innovations"\n- "green hydrogen developments"\n- "renewable energy policy updates"\n\n**Reliable Sources to Check:**\n- **News & Analysis:** Reuters Energy, Bloomberg New Energy Finance, Greentech Media, Energy Storage News\n- **Industry Publications:** Renewable Energy World, PV Magazine, Wind Power Engineering\n- **Research Organizations:** International Energy Agency (IEA), National Renewable Energy Laboratory (NREL)\n- **Government Sources:** Department of Energy websites, EPA clean energy updates\n\n**Specific Topics to Explore:**\n- Perovskite and next-gen solar cells\n- Offshore wind expansion\n- Grid-scale battery storage\n- Green hydrogen production\n- Carbon capture technologies\n- Smart grid innovations\n- Energy policy changes and incentives...', type='text')]
        ```
      </Step>
    </Steps>
  </Tab>

  <Tab title="TypeScript">
    <Steps>
      <Step title="设置您的 API 密钥">
        从 [Claude Console](/settings/keys) 获取您的 API 密钥并将其设置为环境变量：

        ```bash
        export ANTHROPIC_API_KEY='your-api-key-here'
        ```
      </Step>

      <Step title="安装 SDK">
        安装 Anthropic TypeScript SDK：

        ```bash
        npm install @anthropic-ai/sdk
        ```
      </Step>

      <Step title="创建您的代码">
        将其保存为 `quickstart.ts`：

        ```typescript
        import Anthropic from "@anthropic-ai/sdk";

        async function main() {
          const anthropic = new Anthropic();

          const msg = await anthropic.messages.create({
            model: "claude-sonnet-4-5",
            max_tokens: 1000,
            messages: [
              {
                role: "user",
                content: "What should I search for to find the latest developments in renewable energy?"
              }
            ]
          });
          console.log(msg);
        }

        main().catch(console.error);
        ```
      </Step>

      <Step title="运行您的代码">
        ```bash
        npx tsx quickstart.ts
        ```

        **示例输出：**
        ```javascript
        {
          id: 'msg_01ThFHzad6Bh4TpQ6cHux9t8',
          type: 'message',
          role: 'assistant',
          model: 'claude-sonnet-4-5-20250929',
          content: [
            {
              type: 'text',
              text: 'Here are some effective search strategies to find the latest renewable energy developments:\n\n' +
                '## Search Terms to Use:\n' +
                '- "renewable energy news 2024"\n' +
                '- "clean energy breakthroughs"\n' +
                '- "solar wind technology advances"\n' +
                '- "energy storage innovations"\n' +
                '- "green hydrogen developments"\n' +
                '- "offshore wind projects"\n' +
                '- "battery technology renewable"\n\n' +
                '## Best Sources to Check:\n\n' +
                '**News & Industry Sites:**\n' +
                '- Renewable Energy World\n' +
                '- CleanTechnica\n' +
                '- GreenTech Media (now Wood Mackenzie)\n' +
                '- Energy Storage News\n' +
                '- PV Magazine (for solar)...'
            }
          ],
          stop_reason: 'end_turn',
          usage: {
            input_tokens: 21,
            output_tokens: 302
          }
        }
        ```
      </Step>
    </Steps>
  </Tab>

  <Tab title="Java">
    <Steps>
      <Step title="设置您的 API 密钥">
        从 [Claude Console](/settings/keys) 获取您的 API 密钥并将其设置为环境变量：

        ```bash
        export ANTHROPIC_API_KEY='your-api-key-here'
        ```
      </Step>

      <Step title="安装 SDK">
        将 Anthropic Java SDK 添加到您的项目中。首先在 [Maven Central](https://central.sonatype.com/artifact/com.anthropic/anthropic-java) 上找到当前版本。

        **Gradle：**
        ```groovy
        implementation("com.anthropic:anthropic-java:1.0.0")
        ```

        **Maven：**
        ```xml
        <dependency>
          <groupId>com.anthropic</groupId>
          <artifactId>anthropic-java</artifactId>
          <version>1.0.0</version>
        </dependency>
        ```
      </Step>

      <Step title="创建您的代码">
        将其保存为 `QuickStart.java`：

        ```java
        import com.anthropic.client.AnthropicClient;
        import com.anthropic.client.okhttp.AnthropicOkHttpClient;
        import com.anthropic.models.messages.Message;
        import com.anthropic.models.messages.MessageCreateParams;

        public class QuickStart {
            public static void main(String[] args) {
                AnthropicClient client = AnthropicOkHttpClient.fromEnv();

                MessageCreateParams params = MessageCreateParams.builder()
                        .model("claude-sonnet-4-5-20250929")
                        .maxTokens(1000)
                        .addUserMessage("What should I search for to find the latest developments in renewable energy?")
                        .build();

                Message message = client.messages().create(params);
                System.out.println(message.content());
            }
        }
        ```
      </Step>

      <Step title="运行您的代码">
        ```bash
        javac QuickStart.java
        java QuickStart
        ```

        **示例输出：**
        ```java
        [ContentBlock{text=TextBlock{text=Here are some effective search strategies to find the latest renewable energy developments:

        ## Search Terms to Use:
        - "renewable energy news 2024"
        - "clean energy breakthroughs"  
        - "solar/wind/battery technology advances"
        - "energy storage innovations"
        - "green hydrogen developments"
        - "renewable energy policy updates"

        ## Best Sources to Check:
        - **News & Analysis:** Reuters Energy, Bloomberg New Energy Finance, Greentech Media
        - **Industry Publications:** Renewable Energy World, PV Magazine, Wind Power Engineering
        - **Research Organizations:** International Energy Agency (IEA), National Renewable Energy Laboratory (NREL)
        - **Government Sources:** Department of Energy websites, EPA clean energy updates

        ## Specific Topics to Explore:
        - Perovskite and next-gen solar cells
        - Offshore wind expansion
        - Grid-scale battery storage
        - Green hydrogen production..., type=text}}]
        ```
      </Step>
    </Steps>
  </Tab>
</Tabs>

## 后续步骤

现在您已经进行了第一次 Claude API 请求，是时候探索还有什么其他可能性了：

<CardGroup cols={3}>
  <Card title="使用消息" icon="messages" href="/docs/zh-CN/build-with-claude/working-with-messages">
    了解消息 API 的常见模式。
  </Card>
  <Card title="功能概览" icon="brain" href="/docs/zh-CN/api/overview">
    探索 Claude 的高级功能和能力。
  </Card>
  <Card title="客户端 SDK" icon="code-brackets" href="/docs/zh-CN/api/client-sdks">
    发现 Anthropic 客户端库。
  </Card>
  <Card title="Claude 食谱" icon="chef-hat" href="https://github.com/anthropics/anthropic-cookbook">
    通过交互式 Jupyter 笔记本学习。
  </Card>
</CardGroup>