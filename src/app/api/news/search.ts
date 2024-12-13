import { NextResponse } from 'next/server'
import { client } from '@/lib/elasticsearch'


export async function POST(request: Request) {
  try {
    const body = await request.json()
    const searchQuery = body.query

    const searchBody = {
      size: 10,
      sort: [
        {
          publish_at: "desc"
        }
      ],
      _source: [
        "title",
        "url",
        "image_url",
        "content",
        "publish_at",
        "topic_classification",
        "sentiment",
        "urgency_level",
        "target_audience",
        "affected_region"
      ],
      query: searchQuery
        ? {
            multi_match: {
              query: searchQuery,
              fields: ["content", "title", "description"]
            }
          }
        : {
            match_all: {}
          }
    }

    const result = await client.search({
      index: 'news_jakarta',
      body: searchBody
    })

    return NextResponse.json(result)
  } catch (error) {
    console.error('Search error:', error)
    return NextResponse.json(
      { error: 'Failed to search news' },
      { status: 500 }
    )
  }
}